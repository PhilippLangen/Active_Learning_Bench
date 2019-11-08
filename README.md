# Active Learning Benchmark
This repository provides a tool to test various geometric active learning approaches. Tests are conducted on the Cifar-10 dataset using a small CNN.
To select new samples for labelling all samples are projected into a vector space using their network activations at a selected layer. 
Then one of the available selection methods are used to add a batch of newly labelled samples to the training process.

## Installation
```
pip install -r requirements.txt
```
## Usage/Parameters
Start an experiment with:
```
python active_learning_benchmark.py run --[Parameter=value]
```
Current parameters are:
```
    --labeling_strategy=LABELING_STRATEGY
        Choose a strategy to select new samples. Options: random_sampling, greedy_k_center
    --logfile=LOGFILE
        Filename for the json log created for this run
    --initial_training_size=INITIAL_TRAINING_SIZE
        Number of initially labelled samples
    --batch_size=BATCH_SIZE
        Number of samples per gradient update
    --epochs=EPOCHS
        Number of epochs trained between introduction of new samples
    --budget=BUDGET
        Number of samples added in each iteration
    --iterations=ITERATIONS
        Maximum number of labeling iterations
    --learning_rate=LEARNING_RATE
        Learning rate for training
    --target_layer=TARGET_LAYER
        Layer at which activations are extracted
    --vis=VIS
        Toggle plots visualizing activations in 2-D space using PCA
```
