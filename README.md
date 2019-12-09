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
python active_learning_benchmark.py run --[Parameter] [value]
```
Current parameters are:
```
    --labeling_strategy $LABELING_STRATEGY
        Choose a strategy to select new samples. 
        Options: random_sampling, greedy_k_center, spatial_loss_sampling, low_confidence_sampling
    --logfile $LOGFILE
        Filename for the json log created for this run
    --initial_training_size $INITIAL_TRAINING_SIZE
        Number of initially labelled samples
    --batch_size $BATCH_SIZE
        Number of samples per gradient update
    --budget $BUDGET
        Number of samples added in each iteration
    --iterations $ITERATIONS
        Maximum number of labeling iterations
    --target_layer $TARGET_LAYER
        Layer at which activations are extracted
    --vis $VIS
        Toggle plots visualizing activations in 2-D space using PCA
```
#### Create Job Queues
While testing it can be helpful to queue up multiple test runs at once and then just letting them run on their own.
You can use the job_scheduler.py script to create and run such job queues.

To create or extend a job queue run: 
```
python job_queue.py add_new_jobs --[Parameter] [value]
```
Parameters for this function work mostly like the parameters for a single experiment run, but expect lists of setting 
values instead of single values. The resulting queue will contain jobs for all possible combinations of the provided settings.

```
--json_queue 
Expects a path to an existing or desired queue file. If an existing queue file is provided new jobs will be appended to it.
--strategies "[$sampling_strategies]"
Default: "['random_sampling']"
--initial_training_splits "[$initial_split_sizes]"
Default: "[1000]"
--batch_sizes "[$batch_sizes]"
Default: "[32]"
--target_layers "[$target_layers]"
Default: "[4]"
--budgets "[$budgets]"
Default: "[1000]"
--runs $number_of_runs_per_setting_combination
Number of runs that will be added for each unique setting combination.

Note: The number of iterations is automatically inferred from budgets and inital_training_splits, such that the total 
number of labelled samples by the end of a run does not exceed the TOTAL_BUDGET, set as 40000 samples by default.
```

To start working on a job queue call:
```
python job_queue.py run_queue $json_queue_filepath
```
Experiments will be conducted in sequential order. Once a job is completed the queue file is updated allowing you to 
stop / resume the queue, while keeping track of progress.
You can split a queue file into multiple parts using:
```
python job_queue.py split_workload --json_queue $json_queue_filepath --num_splits $number_of_desired_splits 
                                   --remove_original_file $remove_original_file_bool Default:True
```
The created files are named after the original, having \_part_$part_index appended to their end.


### Evaluation

Start by merging log files of runs with identical settings:
```
python evaluation.py merge
```
You can automatically create a variety of plots for your merged log files.
Filenames and directories will automatically be selected, based on the given data:
```
python evaluation.py single_plot --merged_logfile $merged_log_filename

Optional flags:

--plot_individal_runs
Creates additional plots for each run in your logfile.

--exclude_plot_types "[$unwanted_plot_types]"
Plot types are ['accuracy', 'class distribution', 'class distribution information gain', 'confusion matrix',
 'distribution recall correlation', 'distribution precision correlation', 'distribution accuracy correlation']
```
To compare the results for multiple values of a given setting, use:
```
python evaluation.py multi_plot --examined_setting $setting_in_which_runs_differ 
                                --base_settings $dict_of_remaining_settings or $log_filename_with_desired_settings

Optional flags:

--ignored_settings "[$settings_that_may_differ]"
By default this routine will only use log files that differ exclusively in the examined_setting
log files that additionally differ in any of the ignored settings will still be considered.

--exclude_plot_types "[$unwanted_plot_types]"
Plot types are ['accuracy','class distribution information gain']
```
To quickly get all single setting plots for your runs, use:
```
python evaluation.py single_plot_all 

Optional flags function identically to the single_plot flags.

--plot_individal_runs
Creates additional plots for each run in your logfile.

--exclude_plot_types "[$unwanted_plot_types]"
Plot types are ['accuracy', 'class distribution', 'class distribution information gain', 'confusion matrix',
 'distribution recall correlation', 'distribution precision correlation', 'distribution accuracy correlation']
```

