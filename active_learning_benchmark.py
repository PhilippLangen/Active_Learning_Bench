import json
import uuid
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import resnet

VALIDATION_SET_SIZE = 5000
EVALUATION_BATCH_SIZE = 100
MAX_EPOCHS = 200
LEARNING_RATE = 0.0001
MOMENTUM = 0.9


class SimpleCNN(nn.Module):
    """
    Simple CNN network
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Hook:
    """
    Utility class for forward hook
    """
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.input = None
        self.output = None

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class ActiveLearningBench:
    """
    Testing bench for various geometry based active learning strategies
        :param labeling_strategy: Choose a strategy to select new samples. Options: random_sampling, greedy_k_center
        :param logfile: Filename for the json log created for this run
        :param initial_training_size: Number of initially labelled samples
        :param batch_size: Number of samples per gradient update
        :param budget: Number of samples added in each iteration
        :param iterations: Maximum number of labeling iterations
        :param target_layer: Layer at which activations are extracted
        :param vis: Toggle plots visualizing activations in 2-D space using PCA
    """
    def __init__(self, labeling_strategy="random_sampling", logfile="log", initial_training_size=1000, batch_size=32,
                 budget=1000, iterations=20, target_layer=4, vis=False, model="SimpleCNN", data_augmentation=True):
        self.model_type = model.lower()
        # Set a torch device. GPU if available.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Ensure all output directories have been created.
        if not Path('./datasets').is_dir():
            Path('./datasets').mkdir()
        if not Path('./plots').is_dir():
            Path('./plots').mkdir()
        if not Path('./logs').is_dir():
            Path('./logs').mkdir()
        self.logfile = logfile
        self.data_augmentation = data_augmentation
        # normalize data with mean and standard deviation of the dataset.
        if self.data_augmentation:
            self.training_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))])
        else:
            self.training_transform = transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                                                              (0.24703223, 0.24348513, 0.26158784))])
        self.testing_transform = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                                                          (0.24703223, 0.24348513, 0.26158784))])
        # load the CIFAR test and training sets.
        self.training_set = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True,
                                                         transform=self.training_transform)
        self.test_set = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True,
                                                     transform=self.testing_transform)
        self.labeling_strategy = labeling_strategy
        # Ensure that the number of initially labelled samples does not exceed the number of available samples.
        self.initial_training_size = min(initial_training_size, len(self.training_set.data)-VALIDATION_SET_SIZE)
        self.batch_size = batch_size
        self.budget = budget
        self.target_layer = target_layer
        self.vis = vis
        if vis:
            # create base path for vector representation
            self.visualization_base_path = Path.joinpath(Path("plots"),
                                                         Path("vector_visualization"),
                                                         Path(self.logfile))
            run_idx = 1
            while self.visualization_base_path.is_dir():
                self.visualization_base_path = Path.joinpath(Path("plots"),
                                                             Path("vector_visualization"),
                                                             Path(f"{self.logfile}_{run_idx}"))
            self.visualization_base_path.mkdir(parents=True)
        # make sure we don't keep labelling samples if we run out of samples.
        self.iterations = min(iterations,
                              int((len(self.training_set.data)-self.initial_training_size - VALIDATION_SET_SIZE)
                                  / self.budget))
        # keep track of indices of labelled and unlabelled samples
        self.unlabelled_idx = np.arange(0, self.training_set.data.shape[0])
        # select samples for the validation set.
        validation_idx = np.random.choice(self.unlabelled_idx, VALIDATION_SET_SIZE, replace=False)
        self.unlabelled_idx = np.setdiff1d(self.unlabelled_idx, validation_idx)
        # pick initial labelled indices.
        self.labelled_idx = np.random.choice(self.unlabelled_idx, self.initial_training_size, replace=False)
        self.unlabelled_idx = np.setdiff1d(self.unlabelled_idx, self.labelled_idx)
        # create dataloaders for the labelled samples and the validation set.
        labelled_sampler = torch.utils.data.SubsetRandomSampler(self.labelled_idx)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.training_set,
                                                        batch_size=self.batch_size,
                                                        sampler=labelled_sampler, num_workers=2)
        validation_sampler = torch.utils.data.SubsetRandomSampler(validation_idx)
        self.validation_loader = torch.utils.data.DataLoader(dataset=self.training_set,
                                                             batch_size=EVALUATION_BATCH_SIZE,
                                                             sampler=validation_sampler,
                                                             num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_set,
                                                       batch_size=EVALUATION_BATCH_SIZE,
                                                       num_workers=2)
        # create a dataloader, that keeps the initial order of the dataset, so we can connect network outputs with
        # specific input indices
        seq_sampler = torch.utils.data.SequentialSampler(self.training_set)
        self.seq_data_loader = torch.utils.data.DataLoader(dataset=self.training_set,
                                                           batch_size=EVALUATION_BATCH_SIZE,
                                                           sampler=seq_sampler, num_workers=2)

    def get_trained_model(self):
        """
        Trains a model on all labelled samples until validation loss stops decreasing
        :return: trained model
        """
        # Initialize model and optimization tools.
        if self.model_type == "simplecnn":
            model = SimpleCNN()
        elif self.model_type == "resnet18":
            model = resnet.ResNet18()
        else:
            print(f"Unrecognized model type {self.model_type}!")
            raise SystemExit
        model.to(self.device)
        model_path = Path(f"model_{uuid.uuid4().hex}")
        while model_path.exists():
            model_path = Path(f"model_{uuid.uuid4().hex}")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=5e-4)
        min_validation_loss = np.inf
        best_epoch = -1
        print(f"Training on {self.labelled_idx.shape[0]} samples.")
        for epoch in range(MAX_EPOCHS):
            # training
            model.train()
            running_loss = 0.0
            for i, batch in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 20 == 19:  # print every 20 mini-batches
                    print(f'[{epoch}: {i + 1}/{len(self.train_loader)}] loss: {running_loss / 20:.3f}')
                    running_loss = 0.0
            # validation
            model.eval()
            with torch.no_grad():
                average_validation_loss = 0.0
                for i, batch in enumerate(self.validation_loader, 0):
                    inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                    outputs = model(inputs)
                    average_validation_loss += criterion(outputs, labels)
                average_validation_loss /= (VALIDATION_SET_SIZE / EVALUATION_BATCH_SIZE)
                print(f"Validation loss: {average_validation_loss:.3f}")
            # check for validation loss improvement
            if average_validation_loss < min_validation_loss:
                min_validation_loss = average_validation_loss
                best_epoch = epoch
                # create checkpoint of model with lowest validation loss so far
                torch.save(model.state_dict(), model_path)
        print(f"Training completed. Best epoch was {best_epoch} with validation loss {min_validation_loss:.3f}")
        model.load_state_dict(torch.load(model_path))
        model_path.unlink()
        return model, criterion

    def test(self, model):
        """
        testing loop
        :return: accuracy, confusion_matrix
        """
        model.eval()
        with torch.no_grad():
            # initialize statistics containers
            correct_per_class = np.zeros(len(self.test_set.classes))
            total_per_class = np.zeros(len(self.test_set.classes))
            confusion_matrix = torch.zeros(size=(len(self.test_set.classes), len(self.test_set.classes)))
            for batch in self.test_loader:
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                for label in labels:
                    total_per_class[label] += 1
                for tup in torch.stack([predicted, labels], dim=1):
                    confusion_matrix[tup[0]][tup[1]] += 1
                    if tup[0] == tup[1]:
                        correct_per_class[tup[0]] += 1
            accuracy_all = np.sum(correct_per_class)/np.sum(total_per_class)
        print(f'Accuracy of the network on the {len(self.test_set.data)} test images: {100 * accuracy_all:.2f}%')
        confusion_matrix /= len(self.test_set.data)
        return accuracy_all, confusion_matrix.tolist()

    def greedy_k_center(self, activations, losses, confidences):
        """
        Greedy k center strategy
        :param activations: network activations of hooked layer
        :param losses: not used
        :param confidences: not used
        :return:
        """
        # get activations of labelled and unlabelled samples
        unlabelled_data = activations[self.unlabelled_idx]
        labelled_data = activations[self.labelled_idx]
        # create distance matrix: axis 0 is unlabelled data 1 is labelled data
        distance_matrix = torch.cdist(unlabelled_data, labelled_data)
        # for each unlabelled sample find the minimal distance to any labelled sample
        min_dist, _ = torch.min(distance_matrix, dim=1)
        for i in range(self.budget):
            # find index of the sample with the largest minimal distance to any labelled sample
            idx = torch.argmax(min_dist).to("cpu").item()
            # get idx in original dataset order, which will be appended to the labelled_idx list
            new_sample_idx = self.unlabelled_idx[idx]
            # get data of new labelled sample
            new_sample_data = unlabelled_data[idx]
            # delete this index out of unlabelled_idx and add it to labelled_idx,
            # we need to make the same change to our unlabelled data container
            self.unlabelled_idx = np.delete(self.unlabelled_idx, idx)
            self.labelled_idx = np.append(self.labelled_idx, new_sample_idx)
            unlabelled_data = torch.cat((unlabelled_data[0:idx], unlabelled_data[idx + 1:]), dim=0)
            # Finally also delete the corresponding entry from the minimal distance list
            min_dist = torch.cat((min_dist[0:idx], min_dist[idx + 1:]), dim=0)
            # now we need to check if labelling has minimized any minimal distances to labelled samples
            # 3.) finally calc min over 2 values
            # catch edge case - no more unlabelled data -> cant calculate distance to unlabelled data
            if unlabelled_data.size()[0] > 0:
                min_dist, _ = \
                    torch.min(
                        # 2.) add distances to new sample to old min distance_matrix values shape(x,1) -> (x,2)
                        torch.cat((
                            torch.reshape(min_dist, (-1, 1)),
                            # 1.) first calculate distance from unlabelled to new sample
                            torch.cdist(unlabelled_data, torch.reshape(new_sample_data, (1, -1)))),
                            dim=1),
                        dim=1)

        # update data loader
        labelled_sampler = torch.utils.data.SubsetRandomSampler(self.labelled_idx)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.training_set, batch_size=self.batch_size,
                                                        sampler=labelled_sampler, num_workers=2)

    def spatial_loss_sampling(self, activations, losses, confidences):
        """
        Calculates a predicted loss for unlabelled samples, based on losses of surrounding labelled samples.
        The selects samples for labelling, favoring samples with high predictes loss.
        :param activations: activations at hooked layer
        :param losses: losses (only uses losses of labelled samples)
        :param confidences: not used
        :return:
        """
        with torch.no_grad():
            # get activations of unlabelled and labelled samples.
            unlabelled_data = activations[self.unlabelled_idx]
            labelled_data = activations[self.labelled_idx]
            # get losses for labelled samples.
            losses = losses[self.labelled_idx]
            losses = losses.squeeze()
            # create distance matrix axis 0 is unlabelled data 1 is labelled data
            distance_matrix = torch.cdist(unlabelled_data, labelled_data)
            # calculate inverted squared distances. Later used as weights for surrounding losses.
            distance_matrix.pow_(2)
            distance_matrix.reciprocal_()
            # normalize inverted squared distances, so the loss weights for each samples add up to 1.
            F.normalize(distance_matrix, p=1, dim=1, out=distance_matrix)
            # create predicted loss by weighing neighbouring losses values with their inverted squared distances
            predicted_loss = torch.matmul(distance_matrix, losses)
            # normalize again to get a probability distribution
            F.normalize(predicted_loss, p=1, dim=0, out=predicted_loss)
            predicted_loss = predicted_loss.to("cpu").numpy()
            # randomly draw new samples to label, using the normalized predicted loss as distribution
            added_samples = np.random.choice(self.unlabelled_idx, self.budget, replace=False, p=predicted_loss)
            # add new samples and update the dataloaders accordingly
            self.update_train_loader(added_samples)

    def low_confidence_sampling(self, activations, losses, confidences):
        """
        Selects new samples to label based on the networks certainty for the samples. Network uncertainty is considered
        lowest, if the maximum probability for any class is minimal. Selects the samples the network is most uncertain
        about
        :param activations: not used
        :param losses: not used
        :param confidences: softmaxed output of the last network layer
        :return:
        """
        with torch.no_grad():
            # get certainties for unlabelled samples
            unlabelled_confidences = confidences[self.unlabelled_idx]
            # find highest certainty for any class, for each sample
            unlabelled_max_confidences, _ = torch.max(unlabelled_confidences, dim=1)
            # get indices of k samples with lowest maximum certainty
            _, lowest_max_confidence_indices = torch.topk(unlabelled_max_confidences, self.budget, largest=False)
            # get indices based on original dataset order
            added_samples = self.unlabelled_idx[lowest_max_confidence_indices.to("cpu").numpy()]
            # add new samples and update dataloaders accordingly
            self.update_train_loader(added_samples)

    def random_sampling(self, activations, losses, confidences):
        """
        random sampling strategy
        :param activations: vector data is not used in this strategy
        :return:
        """
        # randomly draw new samples
        added_samples = np.random.choice(self.unlabelled_idx, self.budget, replace=False)
        # add new samples and update dataloaders
        self.update_train_loader(added_samples)

    def update_train_loader(self, added_samples):
        """
        update idx lists and create updated training data loader
        :param added_samples:
        :return:
        """
        # remove from unlabelled_idx
        self.unlabelled_idx = np.setdiff1d(self.unlabelled_idx, added_samples)
        # add to labelled_idx
        self.labelled_idx = np.append(self.labelled_idx, added_samples)
        # update data loader
        labelled_sampler = torch.utils.data.SubsetRandomSampler(self.labelled_idx)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.training_set, batch_size=self.batch_size,
                                                        sampler=labelled_sampler, num_workers=2)

    def hook_and_get_num_features(self, model, layer):
        """
        Creates a hook at the given layer and returns the number of inputs going into said layer.
        :param model: network
        :param layer: target layer
        :return: hook, number of layer inputs
        """
        hook = Hook(layer[1])
        # run random image through network to get output size of the target layer
        num_features = 1
        model(torch.zeros((1, 3, 32, 32)).to(self.device))
        for input_dimension in range(1, len(hook.input[0].size())):
            num_features *= hook.input[0].size()[input_dimension]
        return hook, num_features

    def hook_simple_cnn(self, model):
        """
        Creates hook at target layer on a simpleCNN model
        :param model: simpleCNN model
        :return:
        """
        # install a forward hook at the target layer
        for i, layer in enumerate(model._modules.items()):
            if i == self.target_layer:
                return self.hook_and_get_num_features(model, layer)
        print(f"Target layer {self.target_layer} could not be found in {self.model_type}")
        raise SystemExit

    def hook_resnet(self, model):
        """
        Creates hook at target layer on a ResNet model
        :param model: ResNet model
        :return:
        """
        for outer_layer in model._modules.items():
            if type(outer_layer[1]) == torch.nn.Sequential:
                for inner_layer in outer_layer[1]._modules.items():
                    if type(inner_layer[1]) == resnet.BasicBlock:
                        for basic_block_layer in inner_layer[1]._modules.items():
                            if type(basic_block_layer[1]) == torch.nn.Sequential:
                                for shortcut_layer in basic_block_layer[1]._modules.items():
                                    if f"{outer_layer[0]}_{inner_layer[0]}_{basic_block_layer[0]}_{shortcut_layer[0]}" == self.target_layer:
                                        return self.hook_and_get_num_features(model, shortcut_layer)
                            else:
                                if f"{outer_layer[0]}_{inner_layer[0]}_{basic_block_layer[0]}" == self.target_layer:
                                    return self.hook_and_get_num_features(model, basic_block_layer)
                    else:
                        if f"{outer_layer[0]}_{inner_layer[0]}" == self.target_layer:
                            return self.hook_and_get_num_features(model, inner_layer)
            else:
                if outer_layer[0] == self.target_layer:
                    return self.hook_and_get_num_features(model, outer_layer)
        print(f"Target layer {self.target_layer} could not be found in {self.model_type}")
        raise SystemExit

    def create_vector_rep(self, model, criterion):
        """
        extract activations add target layer, order of samples is preserved
        :return: target activations and loss for each sample
        """
        model.eval()
        if type(model) == SimpleCNN:
            hook, num_features = self.hook_simple_cnn(model)
        elif type(model) == resnet.ResNet:
            hook, num_features = self.hook_resnet(model)
        else:
            print(f"Unknown model type {type(model)}")
            raise SystemExit
        if hook is None:
            print(f"Could not find layer {self.target_layer} in {type(model)}")
            raise SystemExit
        print("Creating vector representation of training set")
        with torch.no_grad():
            # initialize containers for data
            activation_all = torch.empty(size=(0, num_features), device=self.device)
            output_all = torch.empty(size=(0, len(self.training_set.classes)), device=self.device)
            loss_all = torch.empty(size=(len(self.training_set.data), 1), device=self.device)
            # iterate over entire dataset in order
            for i, sample in enumerate(self.seq_data_loader, 0):
                image, label = sample[0].to(self.device), sample[1].to(self.device)
                # run forward, get activations from forward hook
                out = model(image)
                # flatten hooked activations - convolutional layer outputs are not flat by default
                activations = hook.input[0].view(-1, num_features)
                # gather activations of hooked layer
                activation_all = torch.cat((activation_all, activations), 0)
                # gather activations of final layer
                output_all = torch.cat((output_all, out), 0)
                for j in range(out.size()[0]):
                    # force loss function to calculate loss for each sample individually
                    loss_all[i*EVALUATION_BATCH_SIZE+j] = criterion(out[j].unsqueeze(0), label[j].unsqueeze(0))
            # softmax the final network activations to get confidence values for each class
            confidence_all = F.softmax(output_all, dim=1)
        hook.close()
        return activation_all, loss_all, confidence_all

    def visualize(self, activations, loss, iteration):
        """
        toy visualization example. Force activation data into 2-D using PCA. Visualize fraction of labelled/unlabelled
        data. Loss as blob size. Not very meaningful, just to get an idea, what can be done.
        :param activations:
        :param loss:
        :param iteration:
        :return:
        """
        print("Creating Visualization")
        np_act = activations.to("cpu").numpy()
        loss = loss.to("cpu").numpy()
        # l2 normalize each feature of activations
        act_norm = sklearn.preprocessing.normalize(np_act, axis=0)
        # perform pca, allow 2 principal components
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(act_norm)
        # ratio of samples we actually visualize so it is not overloaded
        vis_split = 0.02
        # randomly choose subset of labelled/unlabelled data
        labelled_vis_idx = np.random.choice(self.labelled_idx,
                                            int(len(self.labelled_idx)*vis_split), replace=False)
        unlabelled_vis_idx = np.random.choice(self.unlabelled_idx,
                                              int(len(self.unlabelled_idx) * vis_split), replace=False)
        # get pca/ loss data from selected samples
        labelled_vec = principal_components[labelled_vis_idx]
        labelled_loss = loss[labelled_vis_idx]
        unlabelled_vec = principal_components[unlabelled_vis_idx]
        unlabelled_loss = loss[unlabelled_vis_idx]
        # change shape x,2 -> 2,x for scatter syntax
        labelled_vec = np.swapaxes(labelled_vec, 0, 1)
        unlabelled_vec = np.swapaxes(unlabelled_vec, 0, 1)
        # scatter plot, user loss as size
        fig, ax = plt.subplots()
        ax.scatter(unlabelled_vec[0], unlabelled_vec[1], c="grey", s=unlabelled_loss*10, alpha=0.5)
        ax.scatter(labelled_vec[0], labelled_vec[1], c="b", s=labelled_loss*10, alpha=0.5)
        plt.savefig(Path.joinpath(self.visualization_base_path, Path(f"iteration_{iteration}.png")), bbox_inches='tight')
        plt.close(fig)

    def get_class_distribution(self):
        """
        Calculates the distribution of classes in the labelled dataset
        :return: class_distribution
        """
        class_distribution = np.zeros(len(self.training_set.classes))
        for i, batch in enumerate(self.train_loader, 0):
            for label in batch[1]:
                class_distribution[label] += 1
        class_distribution /= len(self.labelled_idx)

        return class_distribution.tolist()

    def run(self):
        """
        run program
        :return:
        """
        # initialize logging lists for various tracked parameters
        accuracy_log = []
        class_distribution_log = []
        confusion_matrix_log = []
        # main loop
        for i in range(self.iterations):
            # fully train model
            model, criterion = self.get_trained_model()
            # benchmark
            accuracy, confusion_matrix = self.test(model)
            accuracy_log.append(accuracy)
            confusion_matrix_log.append(confusion_matrix)
            class_distribution_log.append(self.get_class_distribution())
            # get vector representation
            activations, losses, confidences = self.create_vector_rep(model, criterion)
            if self.vis:
                self.visualize(activations, losses, i)
            # use selected sampling strategy
            print("Select samples for labelling")
            self.__getattribute__(self.labeling_strategy)(activations, losses, confidences)
        # evaluation after last samples have been added
        model, criterion = self.get_trained_model()
        accuracy, confusion_matrix = self.test(model)
        accuracy_log.append(accuracy)
        confusion_matrix_log.append(confusion_matrix)
        class_distribution_log.append(self.get_class_distribution())
        if self.vis:
            activations, losses, confidences = self.create_vector_rep(model, criterion)
            self.visualize(activations, losses, self.iterations)
        # create log file
        log = {'Strategy': self.labeling_strategy, 'Budget': self.budget, 'Initial Split': self.initial_training_size,
               'Iterations': self.iterations, 'Batch Size': self.batch_size, 'Model': self.model_type,
               'Target Layer': self.target_layer, 'Data Augmentation': self.data_augmentation, 'Accuracy': accuracy_log,
               'Class Distribution': class_distribution_log, 'Confusion Matrix': confusion_matrix_log}
        # If the desired filename is already taken, append a number to the filename.
        filepath = Path(f'./logs/{self.logfile}.json')
        i = 1
        while Path(filepath).is_file():
            filepath = Path(f'./logs/{self.logfile}_{i}.json')
            i += 1
        with filepath.open('w', encoding='utf-8') as file:
            json.dump(log, file, ensure_ascii=False)


if __name__ == '__main__':
    fire.Fire(ActiveLearningBench)
