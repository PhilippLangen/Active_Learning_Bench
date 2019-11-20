# Active Learning Benchmark
This repository provides a tool to test various geometric active learning approaches. Tests are conducted on the Cifar-10 dataset using a small CNN.
To select new samples for labelling all samples are projected into a vector space using their network activations at a selected layer. 
Then one of the available selection methods is used to add a batch of newly labelled samples to the training process.

## Installation
```
pip install -r requirements.txt
```
## Usage/Parameters
### Experiments
Start an experiment with:
```
python active_learning_benchmark.py run --[Parameter value]
```
Current parameters are:
```
    --labeling_strategy $LABELING_STRATEGY
        Choose a strategy to select new samples. Options: random_sampling, greedy_k_center
    --logfile $LOGFILE
        Filename for the json log created for this run
    --initial_training_size $INITIAL_TRAINING_SIZE
        Number of initially labelled samples
    --batch_size $BATCH_SIZE
        Number of samples per gradient update
    --epochs $EPOCHS
        Number of epochs trained between introduction of new samples
    --budget $BUDGET
        Number of samples added in each iteration
    --iterations $ITERATIONS
        Maximum number of labeling iterations
    --learning_rate $LEARNING_RATE
        Learning rate for training
    --target_layer $TARGET_LAYER
        Layer at which activations are extracted
    --vis $VIS
        Toggle plots visualizing activations in 2-D space using PCA
```
### Evaluation

Start by merging log files of runs with identical settings:
```
python evaluation.py merge
```
You can automatically create a variety of plots for your merged log files. Filenames and directories will automatically be 
selected, based on the given data:
```
python evaluation.py single_plot --merged_logfile $merged_log_filename

optional flags:

--plot_individal_runs
creates additional plots for each run in your logfile.

--exclude_plot_types [$unwanted_plot_types]
plot types are ['accuracy','class distribution','class distribution information gain','confusion matrix']
```
To compare the results for multiple values of a given setting, use:
```
python evaluation.py multi_plot --examined_setting $setting_in_which_runs_differ --base_settings $dict_of_remaining_settings or $log_filename_with_desired_settings

optional flags:

--ignored_settings "[$settings_that_may_differ]"
by default this routine will only use log files that differ only in the examined_setting
log files that additionally differ in any of the ignored settings will still be considered

--exclude_plot_types "[$unwanted_plot_types]"
plot types are ['accuracy','class distribution information gain']
```
