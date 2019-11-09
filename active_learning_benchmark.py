import json
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
from scipy import spatial
from sklearn.decomposition import PCA


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
        :param epochs: Number of epochs trained between introduction of new samples
        :param budget: Number of samples added in each iteration
        :param iterations: Maximum number of labeling iterations
        :param learning_rate: Learning rate for training
        :param target_layer: Layer at which activations are extracted
        :param vis: Toggle plots visualizing activations in 2-D space using PCA
    """
    def __init__(self, labeling_strategy="random_sampling", logfile="log", initial_training_size=1000, batch_size=32,
                 epochs=5, budget=1000, iterations=20, learning_rate=0.001, target_layer=4, vis=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN()
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        if not Path('./datasets').is_dir():
            Path('./datasets').mkdir()
        if not Path('./plots').is_dir():
            Path('./plots').mkdir()
        if not Path('./logs').is_dir():
            Path('./logs').mkdir()
        self.filepath = Path(f'./logs/{logfile}.json')
        i = 1
        while Path(self.filepath).is_file():
            self.filepath = Path(f'./logs/{logfile}_{i}.json')
            i += 1
        # normalize data with mean/std of dataset
        self.data_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                                                       (0.24703223, 0.24348513, 0.26158784))])
        self.training_set = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True,
                                                         transform=self.data_transform)
        self.test_set = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True,
                                                     transform=self.data_transform)
        self.labeling_strategy = labeling_strategy
        self.initial_training_size = min(initial_training_size, len(self.training_set.data))
        self.batch_size = batch_size
        self.epochs = epochs
        self.budget = budget
        self.target_layer = target_layer
        self.vis = vis
        # make sure we don't keep iterating if we run out of samples
        self.iterations = min(iterations,
                              int((len(self.training_set.data)-self.initial_training_size) / self.budget))
        # keep track of dataset indices and which samples have already been labelled
        self.unlabelled_idx = np.arange(0, self.training_set.data.shape[0])
        # pick initial labelled indices
        self.labelled_idx = np.random.choice(self.unlabelled_idx, self.initial_training_size, replace=False)
        self.unlabelled_idx = np.setdiff1d(self.unlabelled_idx, self.labelled_idx)
        # training sampler only samples out of labelled idx
        labelled_sampler = torch.utils.data.SubsetRandomSampler(self.labelled_idx)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.training_set,
                                                        batch_size=self.batch_size,
                                                        sampler=labelled_sampler, num_workers=2)
        self.no_grad_batch_size = 100   # How many samples are simultaneously processed during testing/act extraction
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_set,
                                                       batch_size=self.no_grad_batch_size,
                                                       num_workers=2)
        # for extraction order of samples needs to be preserved
        seq_sampler = torch.utils.data.SequentialSampler(self.training_set)
        self.seq_data_loader = torch.utils.data.DataLoader(dataset=self.training_set,
                                                           batch_size=self.no_grad_batch_size,
                                                           sampler=seq_sampler, num_workers=2)

    def train(self):
        """
        main training loop
        """
        print(f"Training on {self.labelled_idx.shape[0]} samples.")
        for epoch in range(self.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, batch in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 20 == 19:  # print every 20 mini-batches
                    print(f'[{epoch+1}, {i+1}/{len(self.train_loader)}] loss: {running_loss/20:.3f}')
                    running_loss = 0.0

    def test(self):
        """
        testing loop
        :return: accuracy, confusion_matrix
        """
        correct = 0
        total = 0
        with torch.no_grad():
            confusion_matrix = torch.zeros(size=(len(self.test_set.classes), len(self.test_set.classes)))
            for batch in self.test_loader:
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                for tup in torch.stack([predicted, labels], dim=1):
                    confusion_matrix[tup[0]][tup[1]] += 1
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the {len(self.test_set.data)} test images: {100 * correct / total:.2f}%')
        confusion_matrix /= len(self.test_set.data)
        return correct/total, confusion_matrix.tolist()

    def greedy_k_center(self, activations):                                             # TODO use torch.cdist instead
        """
        greedy k center strategy
        :param activations:
        :return:
        """
        activations = activations.to("cpu").numpy()
        # unlabelled_data , labelled_data are in modified index order
        unlabelled_data = activations[self.unlabelled_idx]
        labelled_data = activations[self.labelled_idx]

        # create distance matrix axis 0 is unlabelled data 1 is labelled data
        distance_matrix = spatial.distance.cdist(unlabelled_data, labelled_data)
        # for each unlabelled date find min distance to labelled
        min_dist = np.min(distance_matrix, axis=1)

        for i in range(self.budget):
            # find index of largest min_dist entry
            idx = np.argmax(min_dist)
            # min_dist is in index modified order, so to find original order index use index list
            new_sample_idx = self.unlabelled_idx[idx]
            # get data of new labelled sample
            new_sample_data = unlabelled_data[idx]
            # we delete this index out of unlabelled_idx and add it to labelled_idx this changes modified idx order
            self.unlabelled_idx = np.delete(self.unlabelled_idx, idx)
            # add the original index of the new sample to the labelled indices
            self.labelled_idx = np.append(self.labelled_idx, new_sample_idx)
            # we also delete the data row, this makes same change to modified idx order balancing out
            unlabelled_data = np.delete(unlabelled_data, idx, axis=0)
            # Finally delete out of distance list, same change to mod idx order
            min_dist = np.delete(min_dist, idx, axis=0)

            # now we need to see if sampling has minimized any min distances to labelled samples
            # finally calc min over 2 values
            min_dist = \
                np.min(
                    # add distances to new sample to old min distance_matrix values shape(x,1) -> (x,2)
                    np.append(
                        np.reshape(min_dist, (-1, 1)),
                        # first calc distance from unlabelled to new sample
                        spatial.distance.cdist(unlabelled_data, np.reshape(new_sample_data, (1, -1))),
                        axis=1),
                    axis=1)

        # update data loader
        labelled_sampler = torch.utils.data.SubsetRandomSampler(self.labelled_idx)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.training_set, batch_size=self.batch_size,
                                                        sampler=labelled_sampler, num_workers=2)

    def random_sampling(self, activations):
        """
        random sampling strategy
        :param activations: vector data is not used in this strategy
        :return:
        """
        added_samples = np.random.choice(self.unlabelled_idx, self.budget, replace=False)
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

    def create_vector_rep(self):
        """
        extract activations add target layer, order of samples is preserved
        :return: target activations and loss for each sample
        """
        hook = None
        num_features = 1
        # install forward hook
        for i, layer in enumerate(self.model._modules.items()):
            if i == self.target_layer:
                hook = Hook(layer[1])
                # run random image through network to get output size of layer
                self.model(torch.zeros((1, 3, 32, 32)).to(self.device))
                for k in range(1, len(hook.output.size())):
                    num_features *= hook.output.size()[k]
        print("Creating vector representation of training set")
        with torch.no_grad():
            # initialize containers for data
            activation_all = torch.empty(size=(0, num_features), device=self.device)
            loss_all = torch.empty(size=(len(self.training_set.data), 1), device=self.device)
            # iterate over entire dataset in order
            for i, sample in enumerate(self.seq_data_loader, 0):
                image, label = sample[0].to(self.device), sample[1].to(self.device)
                # run forward, get activations from forward hook
                out = self.model(image)
                # flatten hooked activations - conv layer outputs are not flat by default
                activations = hook.output.view(-1, num_features)
                # gather activations
                activation_all = torch.cat((activation_all, activations), 0)
                for j in range(out.size()[0]):
                    # force loss_fn to calculate for each sample separately
                    loss_all[i*self.no_grad_batch_size+j] = self.criterion(out[j].unsqueeze(0), label[j].unsqueeze(0))
        hook.close()
        return activation_all, loss_all

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
        plt.savefig(f"./plots/{iteration}.png", bbox_inches='tight')
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
        accuracy = list()
        class_distribution = list()
        confusion_matrix = list()
        # main loop
        for i in range(self.iterations):
            # update model
            self.train()
            # benchmark
            acc, confuse = self.test()
            accuracy.append(acc)
            confusion_matrix.append(confuse)
            class_distribution.append(self.get_class_distribution())
            # get vec representation
            act, loss = self.create_vector_rep()
            if self.vis:
                self.visualize(act, loss, i)
            # use selected sampling strategy
            print("Select samples for labelling")
            self.__getattribute__(self.labeling_strategy)(act)
        # evaluation after last samples have been added
        self.train()
        acc, confuse = self.test()
        accuracy.append(acc)
        confusion_matrix.append(confuse)
        class_distribution.append(self.get_class_distribution())
        if self.vis:
            act, loss = self.create_vector_rep()
            self.visualize(act, loss, self.iterations)
        # create log file
        log = {'Strategy': self.labeling_strategy, 'Budget': self.budget, 'Initial Split': self.initial_training_size,
               'Epochs': self.epochs, 'Iterations': self.iterations, 'Batch Size': self.batch_size,
               'Learning Rate': self.learning_rate, 'Target Layer': self.target_layer, 'Accuracy': accuracy,
               'Class Distribution': class_distribution, 'Confusion Matrix': confusion_matrix}
        with self.filepath.open('w', encoding='utf-8') as file:
            json.dump(log, file, ensure_ascii=False)
            file.close()


if __name__ == '__main__':
    fire.Fire(ActiveLearningBench)
