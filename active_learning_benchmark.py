import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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
    def __init__(self, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN()
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9)
        if not os.path.isdir('./datasets'):
            os.mkdir('./datasets')
        if not os.path.isdir('./plots'):
            os.mkdir('./plots')
        # normalize data with mean/std of dataset
        self.data_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                                                       (0.24703223, 0.24348513, 0.26158784))])
        self.training_set = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True,
                                                         transform=self.data_transform)
        self.test_set = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True,
                                                     transform=self.data_transform)
        self.sampling_strategy = args.labeling_strategy
        self.initial_training_size = args.initial_training_size
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.budget = args.budget
        self.target_layer = args.target_layer
        # make sure we don't keep iterating if we run out of samples
        self.iterations = min(args.iterations,
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
        print("Training on %d samples." % self.labelled_idx.shape[0])
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
                    print('[%d, %d/%d] loss: %.3f' %
                          (epoch + 1, i + 1,  len(self.train_loader), running_loss / 20))
                    running_loss = 0.0

    def test(self):
        """
        testing loop
        :return: accuracy
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.test_loader:
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the %d test images: %d %%' % (len(self.test_set.data),
              100 * correct / total))
        return correct/total

    def random_sampling(self):
        """
        random sampling strategy
        :return: new samples for labelling
        """
        added_samples = np.random.choice(self.unlabelled_idx, self.budget, replace=False)
        return added_samples

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
        num_features = 0
        # install forward hook
        for i, layer in enumerate(self.model._modules.items()):
            if i == self.target_layer:
                num_features = layer[1].out_features
                hook = Hook(layer[1])
        print("creating vector representation of training set")
        with torch.no_grad():
            # initialize containers for data
            activation_all = torch.empty(size=(0, num_features), device=self.device)
            loss_all = torch.empty(size=(len(self.training_set.data), 1), device=self.device)
            # iterate over entire dataset in order
            for i, sample in enumerate(self.seq_data_loader, 0):
                image, label = sample[0].to(self.device), sample[1].to(self.device)
                # run forward, get activations from forward hook
                out = self.model(image)
                activations = hook.output
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
        plt.savefig("./plots/%s.png" % iteration, bbox_inches='tight')
        plt.close(fig)

    def run(self):
        """
        run program
        :return:
        """
        # main loop
        for i in range(self.iterations):
            # update model
            self.train()
            # benchmark
            self.test()
            # get vec representation
            act, loss = self.create_vector_rep()
            self.visualize(act, loss, i)
            # use selected sampling strategy
            added_samples = self.__getattribute__(self.sampling_strategy)()
            # add samples to labelled_idx
            self.update_train_loader(added_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=5, type=int, help="number of epochs to train")
    parser.add_argument('-batch_size', default=32, type=int, help="number of samples per optimization step")
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help="learning rate")
    parser.add_argument('-init_size', '--initial_training_size', default=10000, type=int,
                        help="initial number of labelled training samples")
    parser.add_argument('-target_layer', default=4, type=int, help="layer where activations are extracted")
    parser.add_argument('-budget', default=1000, type=int,
                        help="number of samples added after each active learning iteration")
    parser.add_argument('-iterations', default=40, type=int,
                        help="number of active learning cycles - will not continue if entire dataset is labelled")
    parser.add_argument('-ls', '--labeling_strategy', default='random_sampling',
                        help="strategy to choose unlabelled samples, options: random_sampling")

    arguments = parser.parse_args()
    print(arguments)
    ActiveLearningBench(arguments).run()
