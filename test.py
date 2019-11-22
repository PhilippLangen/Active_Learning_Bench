import unittest

import numpy as np
import torch
from scipy import spatial


class FrameworkDummy:

    def __init__(self):
        self.unlabelled_idx = np.asarray([0, 2, 3, 4, 7, 8, 9])
        self.labelled_idx = np.asarray([5, 1, 6])
        self.activations = torch.tensor([[1., 9.],
                                         [2., 4.],
                                         [4., 8.],
                                         [5., 2.],
                                         [5., 5.],
                                         [7., 3.],
                                         [7., 7.],
                                         [9., 9.],
                                         [9., 3.],
                                         [10., 7.]]).to("cuda:0")
        self.loss = torch.tensor([3., 2., 3., 2., 4., 1., 2., 2., 1., 3.]).to("cuda:0")
        self.loss = self.loss.unsqueeze(dim=1)
        self.budget = 4

    def greedy_k_center_scipy(self, activations):
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

    def greedy_k_center_torch(self, activations):
        """
        greedy k center strategy
        :param activations:
        :return:
        """
        # unlabelled_data , labelled_data are in modified index order
        unlabelled_data = activations[self.unlabelled_idx]
        labelled_data = activations[self.labelled_idx]
        # create distance matrix axis 0 is unlabelled data 1 is labelled data
        distance_matrix = torch.cdist(unlabelled_data, labelled_data)
        # for each unlabelled date find min distance to labelled
        min_dist, _ = torch.min(distance_matrix, dim=1)
        for i in range(self.budget):
            # find index of largest min_dist entry
            idx = torch.argmax(min_dist).to("cpu").item()
            # min_dist is in index modified order, so to find original order index use index list
            new_sample_idx = self.unlabelled_idx[idx]
            # get data of new labelled sample
            new_sample_data = unlabelled_data[idx]
            # we delete this index out of unlabelled_idx and add it to labelled_idx this changes modified idx order
            self.unlabelled_idx = np.delete(self.unlabelled_idx, idx)
            # add the original index of the new sample to the labelled indices
            self.labelled_idx = np.append(self.labelled_idx, new_sample_idx)
            # we also delete the data row, this makes same change to modified idx order balancing out
            unlabelled_data = torch.cat((unlabelled_data[0:idx],unlabelled_data[idx+1:]), dim=0)
            # Finally delete out of distance list, same change to mod idx order
            min_dist = torch.cat((min_dist[0:idx], min_dist[idx+1:]), dim=0)
            # now we need to see if sampling has minimized any min distances to labelled samples
            # finally calc min over 2 values
            min_dist, _ = \
                torch.min(
                    # add distances to new sample to old min distance_matrix values shape(x,1) -> (x,2)
                    torch.cat((
                        torch.reshape(min_dist, (-1, 1)),
                        # first calc distance from unlabelled to new sample
                        torch.cdist(unlabelled_data, torch.reshape(new_sample_data, (1, -1)))),
                        dim=1),
                    dim=1)

    def spatial_loss_sampling(self, activations, loss):
        loss = loss[self.labelled_idx]
        loss = loss.squeeze()
        unlabelled_data = activations[self.unlabelled_idx]
        labelled_data = activations[self.labelled_idx]
        # create distance matrix axis 0 is unlabelled data 1 is labelled data
        distance_matrix = torch.cdist(unlabelled_data, labelled_data)
        # use inverted squared distance to strongly favor close samples
        print(distance_matrix)
        distance_matrix.pow_(2)
        print(distance_matrix)
        distance_matrix.reciprocal_()
        print(distance_matrix)
        # normalize distances so samples with near neighbours do not get higher loss automatically
        torch.nn.functional.normalize(distance_matrix, p=1, dim=1, out=distance_matrix)
        print(distance_matrix)
        # create predicted loss by weighing neighbouring loss values with distance
        predicted_loss = torch.matmul(distance_matrix, loss)
        # normalize to get probability distribution
        predicted_loss = torch.nn.functional.normalize(predicted_loss, p=1, dim=0).to("cpu").numpy()
        # get new samples be randomly drawing with
        added_samples = np.random.choice(self.unlabelled_idx, self.budget, replace=False, p=predicted_loss)
        print(added_samples)

        # update train loader form here

        self.unlabelled_idx = np.setdiff1d(self.unlabelled_idx, added_samples)
        # add to labelled_idx
        self.labelled_idx = np.append(self.labelled_idx, added_samples)
        print(self.unlabelled_idx)
        print(self.labelled_idx)


class KCenterTest(unittest.TestCase):

    def test_scipy_impl(self):
        fd = FrameworkDummy()
        fd.greedy_k_center_scipy(fd.activations)
        self.assertEqual(fd.labelled_idx.tolist(), [5, 1, 6, 0, 2, 9, 4])
        self.assertEqual(fd.unlabelled_idx.tolist(), [3, 7, 8])

    def test_torch_impl(self):
        fd = FrameworkDummy()
        fd.greedy_k_center_torch(fd.activations)
        self.assertEqual(fd.labelled_idx.tolist(), [5, 1, 6, 0, 2, 9, 4])
        self.assertEqual(fd.unlabelled_idx.tolist(), [3, 7, 8])


class SpatialLoss(unittest.TestCase):

    def test_spatial_impl(self):
        fd = FrameworkDummy()
        fd.spatial_loss_sampling(fd.activations, fd.loss)
        # need some test conditions :P it does work though
        # if promising implement iterative version


if __name__ == '__main__':
    unittest.main()
