# Sampling Strategies

There are currently four different sampling strategies implemented in this repository.
These strategies are used to mark a number of new samples for labeling after each completed training iteration.

## Random Sampling 

Random sampling is the baseline strategy we compare against in our experiments. Here new samples for labelling are chosen at random.

## Low Confidence Sampling

Low confidence sampling is a basic sampling strategy based on network confidence when classifying unlabelled samples.
This strategy selects the samples the network is most uncertain about. Uncertainty of a sample is inferred from the softmax activations of the
last network layer as `1 - max(softmax_activations)` - Or the lower the confidence in the class the sampled has been classified as, the higher the uncertainty.

An alternative way to define uncertainty is as the entropy of the softmax activations.

## Greedy K Center

Greedy k center is the greedy implementation of the strategy presented in this [paper](https://arxiv.org/abs/1708.00489). This is the first approach using the created vector representation. 
The basic idea is to find a subset of the dataset such that all data points are closely enveloped by a labelled sample, thereby finding a set that generalizes the entire dataset.
Finding an optimal selection is NP-hard, however the paper proposes additional optimizations that provide results closer to the optimal solution and have a strategy to ignore outliers.
More information is provided at the [repository of the authors](https://github.com/ozansener/active_learning_coreset).  

## Spatial Loss Sampling 

Spatial loss sampling aims to map network performance to the spatial domain to select samples in regions where network performance is poor.
A heuristic for network performance issues at a given point is calculated as the cross entropy loss of all labelled samples weighted with their inverted squared distance to said point. 
These weighting factors are also normalized to ensure points with many close neighbours do not get favored over points in less dense areas.

$$
w_i = \frac {1}{\|}
$$
## Implementing New Strategies
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTI0NjMyMzMwOF19
-->