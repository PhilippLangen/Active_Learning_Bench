import unittest

import numpy as np
import torch
from scipy import spatial


class FrameworkDummy:

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                                         [10., 7.]]).to(self.device)
        self.confidences = torch.tensor(
                [[0.1174, 0.0715, 0.0753, 0.1045, 0.0718, 0.0887, 0.0730, 0.0727, 0.1530,   # 17.22 u 7 - 0
                 0.1722],
                 [0.1095, 0.1520, 0.0683, 0.0779, 0.0717, 0.1250, 0.0746, 0.0753, 0.1223,   # 12.50
                 0.1234],
                 [0.0793, 0.0937, 0.0956, 0.1133, 0.1153, 0.0863, 0.0760, 0.1620, 0.0811,   # 16.20 u 6 - 2
                 0.0975],
                 [0.0604, 0.1313, 0.0800, 0.1207, 0.0806, 0.1187, 0.1077, 0.0603, 0.0948,   # 14.54 u 2 - 3
                 0.1454],
                 [0.0931, 0.1574, 0.0630, 0.1146, 0.0795, 0.0738, 0.0943, 0.1475, 0.0775,   # 15.74 u 5 - 4
                 0.0994],
                 [0.0657, 0.1229, 0.1163, 0.1013, 0.1116, 0.1130, 0.0734, 0.1187, 0.1113,   # 12.29
                 0.0658],
                 [0.1057, 0.0817, 0.1101, 0.1016, 0.0917, 0.0790, 0.1516, 0.0629, 0.1017,   # 15.16
                 0.1139],
                 [0.0877, 0.1135, 0.0720, 0.1558, 0.0699, 0.1171, 0.0759, 0.1413, 0.0887,   # 15.58 u 4 - 7
                 0.0782],
                 [0.1192, 0.1312, 0.1439, 0.0960, 0.0786, 0.1291, 0.0753, 0.0724, 0.0808,   # 14.39 u 1 - 8
                 0.0735],
                 [0.1377, 0.0848, 0.0984, 0.0763, 0.0638, 0.1365, 0.1018, 0.1489, 0.0597,   # 14.89 u 3 - 9
                 0.0921]]).to(self.device)
        self.loss = torch.tensor([3., 2., 3., 2., 4., 1., 2., 2., 1., 3.]).to(self.device)
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
        print(predicted_loss)
        added_samples = np.random.choice(self.unlabelled_idx, self.budget, replace=False, p=predicted_loss)
        print(added_samples)

        # update train loader form here

        self.unlabelled_idx = np.setdiff1d(self.unlabelled_idx, added_samples)
        # add to labelled_idx
        self.labelled_idx = np.append(self.labelled_idx, added_samples)
        print(self.unlabelled_idx)
        print(self.labelled_idx)

    def low_confidence_sampling(self, confidences):
        with torch.no_grad():
            unlabelled_confidences = confidences[self.unlabelled_idx]
            print(unlabelled_confidences)
            unlabelled_max_confidences, _ = torch.max(unlabelled_confidences, dim=1)
            print(unlabelled_max_confidences)
            _, lowest_max_confidence_indices = torch.topk(unlabelled_max_confidences, self.budget, largest=False)
            added_samples = self.unlabelled_idx[lowest_max_confidence_indices]
            print(added_samples)

            self.unlabelled_idx = np.setdiff1d(self.unlabelled_idx, added_samples)
            self.labelled_idx = np.append(self.labelled_idx, added_samples)


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


class SpatialLossTest(unittest.TestCase):

    def test_spatial_impl(self):
        fd = FrameworkDummy()
        fd.spatial_loss_sampling(fd.activations, fd.loss)
        # need some test conditions :P it does work though
        # if promising implement iterative version


class UncertaintyTest(unittest.TestCase):

    def test_uncertainty_impl(self):
        fd = FrameworkDummy()
        fd.low_confidence_sampling(fd.confidences)
        self.assertEqual(fd.unlabelled_idx.tolist(), [0, 2, 4])
        self.assertCountEqual(fd.labelled_idx.tolist(), [1, 3, 5, 6, 7, 8, 9])


if __name__ == '__main__':
    unittest.main()
