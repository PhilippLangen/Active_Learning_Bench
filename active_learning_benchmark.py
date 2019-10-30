import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, ext=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if ext:
            hooked = x
        x = self.fc3(x)
        if ext:
            return x, hooked
        return x


class ActiveLearningBench:
    def __init__(self, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN()
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9)
        if not os.path.isdir('./datasets'):
            os.mkdir('./datasets')
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
        self.iterations = min(args.iterations,
                              int((len(self.training_set.data)-self.initial_training_size) / self.budget))
        self.unlabelled_idx = np.arange(0, self.training_set.data.shape[0])
        self.labelled_idx = np.random.choice(self.unlabelled_idx, self.initial_training_size, replace=False)
        self.unlabelled_idx = np.setdiff1d(self.unlabelled_idx, self.labelled_idx)
        self.labelled_sampler = torch.utils.data.SubsetRandomSampler(self.labelled_idx)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.training_set, batch_size=self.batch_size,
                                                        sampler=self.labelled_sampler, num_workers=2)
        self.no_grad_batch_size = 100
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_set, batch_size=self.no_grad_batch_size, num_workers=2)
        seq_sampler = torch.utils.data.SequentialSampler(self.training_set)
        self.seq_data_loader = torch.utils.data.DataLoader(dataset=self.training_set, batch_size=self.no_grad_batch_size, sampler=seq_sampler,
                                                      num_workers=2)

    def train(self):
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
        added_samples = np.random.choice(self.unlabelled_idx, self.budget, replace=False)
        self.update_train_loader(added_samples)

    def update_train_loader(self, added_samples):
        self.unlabelled_idx = np.setdiff1d(self.unlabelled_idx, added_samples)
        self.labelled_idx = np.append(self.labelled_idx, added_samples)
        self.labelled_sampler = torch.utils.data.SubsetRandomSampler(self.labelled_idx)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.training_set, batch_size=self.batch_size,
                                                        sampler=self.labelled_sampler, num_workers=2)

    def create_vector_rep(self):
        print("creating vector representation of training set")
        with torch.no_grad():
            vector_map = dict()
            for i, sample in enumerate(self.seq_data_loader, 0):
                image, label = sample[0].to(self.device), sample[1].to(self.device)
                out, activations = self.model(image, True)
                np_activations = activations.to("cpu").numpy()
                for j in range(out.size()[0]):
                    entry = {"loss": self.criterion(out[j].unsqueeze(0), label[j].unsqueeze(0)).to("cpu").numpy(), "vector": np_activations[j], "labelled": False}
                    vector_map[i*self.no_grad_batch_size+j] = entry
            for idx in self.labelled_idx:
                vector_map[idx]["labelled"] = True
            return vector_map

    def run(self):
        for i in range(self.iterations):
            self.train()
            self.test()
            self.create_vector_rep()
            self.__getattribute__(self.sampling_strategy)()

    def test_data_loaders(self):
        seq_sampler = torch.utils.data.SequentialSampler(self.training_set)
        seq_data_loader = torch.utils.data.DataLoader(dataset=self.training_set, batch_size=2, sampler=seq_sampler, num_workers=2)
        for i, sample in enumerate(seq_data_loader, 0):
            loader = sample
            direct = self.training_set.data[2*i]
            direct2 =self.training_set.data[2*i+1]
            direct = self.data_transform(direct)
            direct2 = self.data_transform(direct2)
            print(loader[0][0])
            print("%%%%")
            print(loader[0][1])
            print(loader[0].size())
            print("_")
            print(len(direct))
            print(direct[0].size())
            print(direct[0][0])
            print(direct[1][0])
            print(direct[2][0])
            print("####")
            print(direct2[0][0])
            print(direct2[1][0])
            print(direct2[2][0])
            print("=================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=5, type=int, help="number of epochs to train")
    parser.add_argument('-batch_size', default=32, type=int, help="number of samples per optimization step")
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help="learning rate")
    parser.add_argument('-init_size', '--initial_training_size', default=10000, type=int,
                        help="initial number of labelled training samples")
    parser.add_argument('-budget', default=1000, type=int,
                        help="number of samples added after each active learning iteration")
    parser.add_argument('-iterations', default=40, type=int,
                        help="number of active learning cycles - will not continue if entire dataset is labelled")
    parser.add_argument('-ls', '--labeling_strategy', default='random_sampling',
                        help="strategy to choose unlabelled samples, options: random_sampling")

    arguments = parser.parse_args()
    print(arguments)
    ActiveLearningBench(arguments).run()
