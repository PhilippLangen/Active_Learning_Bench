import json
from collections import defaultdict
from pathlib import Path

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

PLOT_PADDING_FACTOR = 40
CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MERGED_LOGS_PATH = "./logs/merged_logs"


def merge_similar_runs():
    """
    Combine results of runs with identical settings. This function will automatically group log files by their settings
    and creates a merged log for each unique setting combination.
    :return:
    """
    # create/ensure existence of log directories
    target_path_base = MERGED_LOGS_PATH
    source_path_base = "./logs"
    if not Path(source_path_base).is_dir():
        print(f"Logfile directory {source_path_base} not found! Run experiments before using the evaluation tool")
        raise SystemExit
    if not Path(target_path_base).is_dir():
        Path(target_path_base).mkdir()
    source_path_base = Path(source_path_base)
    # find all json files in ./logs
    log_files = list(source_path_base.glob('*.json'))
    log_map = defaultdict(lambda: [])
    # create list of logs for each unique settings key
    for idx, log in enumerate(log_files):
        with log.open() as json_file:
            json_data = json.load(json_file)
            key = (json_data["Strategy"], json_data["Budget"], json_data["Initial Split"],
                   json_data["Iterations"], json_data["Batch Size"], json_data["Target Layer"], json_data["Model"],
                   json_data["Data Augmentation"])
            log_map[key].append(json_data)
    # loop over each setting key
    for key, log_list in log_map.items():
        # initialize data containers
        acc = np.empty(shape=(0, key[3]+1))
        class_dist = np.empty(shape=(0, key[3]+1, 10))
        conf_mat = np.empty(shape=(0, key[3]+1, 10, 10))
        info_gain = np.empty(shape=(0, key[3]+1))
        # accumulate date over multiple runs
        for log in log_list:

            acc = np.append(acc, np.expand_dims(np.asarray(log["Accuracy"]), axis=0), axis=0)
            class_dist = np.append(class_dist, np.expand_dims(np.asarray(log["Class Distribution"]), axis=0), axis=0)
            conf_mat = np.append(conf_mat, np.expand_dims(np.asarray(log["Confusion Matrix"]), axis=0), axis=0)
            info_gain = np.append(info_gain,
                                  np.asarray([scipy.stats.entropy(x, np.ones(10)/10)
                                              for x in np.asarray(log["Class Distribution"])]).reshape((1, -1)), axis=0)
        # create mean and standard deviation info over runs
        acc_mean = np.mean(acc, axis=0)
        class_dist_mean = np.mean(class_dist, axis=0)
        acc_std = np.std(acc, axis=0)
        class_dist_std = np.std(class_dist, axis=0)
        conf_mat_mean = np.mean(conf_mat, axis=0)
        conf_mat_std = np.std(conf_mat, axis=0)
        info_gain_mean = np.mean(info_gain, axis=0)
        info_gain_std = np.std(info_gain, axis=0)
        # turn back to serializable format
        acc = acc.tolist()
        acc_mean = acc_mean.tolist()
        acc_std = acc_std.tolist()
        class_dist = class_dist.tolist()
        class_dist_mean = class_dist_mean.tolist()
        class_dist_std = class_dist_std.tolist()
        conf_mat = conf_mat.tolist()
        conf_mat_mean = conf_mat_mean.tolist()
        conf_mat_std = conf_mat_std.tolist()
        info_gain = info_gain.tolist()
        info_gain_mean = info_gain_mean.tolist()
        info_gain_std = info_gain_std.tolist()
        # create json structure
        merged_dict = {"Strategy": key[0], "Budget": key[1], "Initial Split": key[2],
                       "Iterations": key[3], "Batch Size": key[4], "Target Layer": key[5], "Model": key[6],
                       "Data Augmentation": key[7], "Accuracy All": acc,
                       "Accuracy Mean": acc_mean, "Accuracy Std": acc_std,
                       "Class Distribution All": class_dist, "Class Distribution Mean": class_dist_mean,
                       "Class Distribution Std": class_dist_std, "Confusion Matrix All": conf_mat,
                       "Confusion Matrix Mean": conf_mat_mean, "Confusion Matrix Std": conf_mat_std,
                       "Information Gain All": info_gain, "Information Gain Mean": info_gain_mean,
                       "Information Gain Std": info_gain_std}
        # generate a filename by settings
        target_file = Path(f"{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}_{key[5]}_{key[6]}"
                           f"{'_data_augmentation' if key[7] else ''}.json")
        # create json file
        with Path.joinpath(Path(target_path_base, target_file)).open('w', encoding='utf-8') as file:
            json.dump(merged_dict, file, ensure_ascii=False)


def create_plots_over_setting(examined_setting, base_setting, ignored_settings=[],  exclude_plot_types=None):
    """
    Creates various plots of runs that differ in exactly one setting, eg. different sampling strategy.
    :param examined_setting: Setting you want to look into. Options: Strategy, Budget, Initial Split,
     Batch size, Iterations, Target Layer
    :param base_setting: Provides the shared settings, that are constant over all considered log files as a dict.\n
    Alternatively can be provided as the name of a log file. Settings will then be inferred from it.
    :param ignored_settings: List of Settings to ignore while gathering log files. Useful if there are dependencies
    between the examined setting and the base settings eg. Budget and Iterations.
    :param exclude_plot_types: list of plot types to omit. Options:["accuracy","class distribution information gain"]
    :return:
    """
    # Find matching log files
    if exclude_plot_types is None:
        exclude_plot_types = {}
    exclude_plot_types = {x.lower() for x in exclude_plot_types}
    plot_base_path = Path("./plots/setting_evaluation_plots")
    allowed_variable_settings = ["Strategy", "Budget", "Initial Split", "Batch Size", "Iterations", "Target Layer",
                                 "Model", "Data Augmentation"]
    examined_setting = examined_setting.title()
    if examined_setting in allowed_variable_settings:
        shared_settings = [setting for setting in allowed_variable_settings if examined_setting != setting]
    else:
        print(f"\'{examined_setting}\' did not match any allowed variable setting!"
              f"\nAllowed settings are: {allowed_variable_settings}")
        raise SystemExit
    if type(base_setting) is dict:
        missing_settings = False
        for setting in shared_settings:
            if setting not in base_setting.keys():
                print(f"{setting} missing from provided base settings!")
                missing_settings = True
        if missing_settings:
            raise SystemExit
        shared_settings_dict = {setting: base_setting[setting] for setting in shared_settings}
    else:
        base_log_path = Path.joinpath(Path(MERGED_LOGS_PATH), Path(base_setting))
        if base_log_path.is_file():
            with base_log_path.open() as json_file:
                base_log = json.load(json_file)
        else:
            print(f"No log found at {base_log_path}!")
            raise SystemExit
        shared_settings = [setting for setting in shared_settings if setting not in ignored_settings]
        print(shared_settings)
        shared_settings_dict = {setting: base_log[setting] for setting in shared_settings}
    if not Path(MERGED_LOGS_PATH).is_dir():
        print("Merged logs directory not found!\n"
              " Make sure your log directory is not empty and you run merge_similar_runs() before creating plots")
        raise SystemExit
    all_log_paths = Path(MERGED_LOGS_PATH).glob('*.json')
    all_log_paths = sorted(all_log_paths)
    all_logs = []
    for log_path in all_log_paths:
        with log_path.open() as json_file:
            all_logs.append(json.load(json_file))
    print(f"Searching for logs matching: {shared_settings_dict},\n that differ in {examined_setting}")
    matching_logs = []
    for log in all_logs:
        matching = True
        for setting in shared_settings:
            if shared_settings_dict[setting] != log[setting]:
                matching = False
        if matching:
            matching_logs.append(log)
    if len(matching_logs) == 0:
        print("There are no logs matching the selected settings")
        raise SystemExit
    elif len(matching_logs) == 1:
        print("There is only one log matching your settings. Use create_single_setting_plots() instead")
        raise SystemExit
    else:
        print(f"Found {len(matching_logs)} logs matching these setting!")
    # create directory
    dir_name = '_'
    dir_name = dir_name.join([str(val) for val in shared_settings_dict.values() if type(val) != bool])
    if shared_settings_dict["Data Augmentation"]:
        dir_name = f"{dir_name}_data_augmentation"
    dir_name = f"Eval:{examined_setting}_with:{dir_name}"
    plot_base_path = Path.joinpath(plot_base_path, Path(dir_name))
    if not plot_base_path.is_dir():
        plot_base_path.mkdir(parents=True)
    # plots
    if "accuracy" not in exclude_plot_types:
        path = Path.joinpath(plot_base_path, Path("Accuracy Plot.png"))
        acc_data_mean = []
        acc_data_error = []
        x_data = []
        labels = []
        for log in matching_logs:
            acc_data_mean.append(np.asarray(log["Accuracy Mean"]))
            acc_data_error.append(np.asarray(log["Accuracy Std"]))
            x_data.append((np.arange(log["Iterations"]+1) * log["Budget"]) + log["Initial Split"])
            labels.append(f"{examined_setting}: {log[examined_setting]}")

        create_line_plot(x_data, acc_data_mean, path, labels, acc_data_error)

    if "class distribution information gain" not in exclude_plot_types:
        path = Path.joinpath(plot_base_path, Path("Information Gain.png"))
        ig_data_mean = []
        ig_data_error = []
        x_data = []
        labels = []
        for log in matching_logs:
            ig_data_mean.append(np.asarray(log["Information Gain Mean"]))
            ig_data_error.append(np.asarray(log["Information Gain Std"]))
            x_data.append((np.arange(log["Iterations"]+1) * log["Budget"]) + log["Initial Split"])
            labels.append(f"{examined_setting}: {log[examined_setting]}")

        create_line_plot(x_data, ig_data_mean, path, labels, ig_data_error,
                         title="Information Gain Over Balanced Distribution - High = Unbalanced",
                         y_label="Information Gain")


def create_line_plot(x_data, y_data, out_path, labels, error=None, x_bounds=None, y_bounds=None,
                     title="Accuracy Plot", x_label="Labelled Samples", y_label="Accuracy",
                     show_legend=True):
    """
    Creates a 2D-line plot. Expects lists of numpy arrays as x,y and error data and a list of labels.
     Each list entry corresponds to a run. Uses error bars if error data has been provided. Error data shape must match
     y_data shape if provided.
    distinct run.
    :param x_data: list of 1-D np.arrays - each np arrays contains x data of one line/run.
    :param y_data: list of 1-D np.arrays - each np arrays contains y data of one line/run.
    :param out_path: desired out_file path. Expects pathlib filepath.
    :param labels: list of labels. Expects a string for each line/run.
    :param error: list of 1-D np.arrays - each np arrays contains error data of one line/run, matches y_data shape.
    :param x_bounds: tuples of minimum and maximum bound of x axis. If None - will be inferred of x_data.
    :param y_bounds: tuples of minimum and maximum bound of y axis. If None - will be inferred of y_data and error.
    :param title: title of plot.
    :param x_label: label on x-axis.
    :param y_label: label on y-axis.
    :param show_legend: toggle whether legend is drawn, default: False.
    :return:
    """
    if x_bounds is None:
        x_bounds = calculate_bounds(x_data)
    if y_bounds is None:
        y_min, y_max = calculate_bounds(y_data)
        if error is not None:
            std_max = max([np.max(err) for err in error])
            y_min -= std_max
            y_max += std_max
        y_bounds = (y_min, y_max)
    fig, ax = plt.subplots(figsize=(7, 4))
    if error is not None:
        for line_idx in range(len(y_data)):
            ax.errorbar(x=x_data[line_idx], y=y_data[line_idx], yerr=error[line_idx], marker='o', markersize=4,
                        label=labels[line_idx], alpha=0.65)
    else:
        for line_idx in range(len(y_data)):
            ax.plot(x_data[line_idx], y_data[line_idx], marker='o', markersize=4, label=labels[line_idx], alpha=0.65)
    if show_legend:
        ax.legend(loc='lower right')
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def create_single_setting_plots(merged_logfile, plot_individual_runs=False, exclude_plot_types=None):
    """
    Creates plots out of a single merged log file, such as accuracy, class distribution plot or confusion matrices.
    :param merged_logfile: Filename of the merged log file to use.
    :param plot_individual_runs: Whether or not plots for individual runs are created
    :param exclude_plot_types: List of plot types that are omitted, Options: ["accuracy","class distribution",
    "class distribution information gain", "confusion matrix", "distribution recall correlation",
    "distribution precision correlation", "distribution accuracy correlation"]
    :return:
    """
    if exclude_plot_types is None:
        exclude_plot_types = {}
    # get log data
    logfile = Path(f"{MERGED_LOGS_PATH}/{merged_logfile}")
    if not logfile.is_file():
        print(f"invalid filename: {logfile} does not exist!")
        raise SystemExit
    with logfile.open() as json_file:
        log_data = json.load(json_file)
    plot_base_path = Path(f"./plots/single_setting_plots/{merged_logfile[:-5]}")
    # create directory for all plots of logfile
    if not plot_base_path.is_dir():
        plot_base_path.mkdir(parents=True)
    exclude_plot_types = {x.lower() for x in exclude_plot_types}
    # mean plots
    # accuracy
    print("Creating averaged plots..")
    if 'accuracy' not in exclude_plot_types:
        print("Creating accuracy plots.")

        y_data = [np.asarray(log_data["Accuracy Mean"])]
        x_data = [(np.arange(y_data[0].shape[0])*log_data["Budget"]) + log_data["Initial Split"]]
        error_data = [np.asarray(log_data["Accuracy Std"])]
        create_line_plot(x_data, y_data, Path.joinpath(plot_base_path, Path("Mean_Accuracy_Plot.png")),
                         error=error_data, y_label="Mean Accuracy", labels=["mean acc"], show_legend=False)
    # class distribution
    if "class distribution" not in exclude_plot_types:
        print("Creating class distribution plots.")
        mean_data = np.asarray(log_data["Class Distribution Mean"])
        error = np.asarray(log_data["Class Distribution Std"])
        create_class_dist_plot(mean_data, Path.joinpath(plot_base_path, Path("Class_Distribution_Mean")),
                               "Mean_Class_Distribution_Plot", error=error, title="Mean Class Distribution",
                               y_label="Mean Class Distribution", labels=["mean_class_dist"], show_legend=False)
    # class distribution information gain
    if "class distribution information gain" not in exclude_plot_types:
        print("Creating information gain plots.")
        y_data = [np.asarray(log_data["Information Gain Mean"])]
        x_data = [(np.arange(y_data[0].shape[0])*log_data["Budget"]) + log_data["Initial Split"]]
        error_data = [np.asarray(log_data["Information Gain Std"])]
        create_line_plot(x_data, y_data, Path.joinpath(plot_base_path,
                                                       Path("Mean_Information_Gain_Plot.png")), error=error_data,
                         labels=["mean_ig"], x_label="Labelled Samples", y_label="Information Gain",
                         title="Information Gain Over Balanced Distribution - High = Unbalanced",
                         show_legend=False)
    if "confusion matrix" not in exclude_plot_types:
        print("Creating confusion matrix plots.")
        confusion_matrix_data = np.asarray(log_data["Confusion Matrix Mean"])
        # normalize over predicted ratio for each class - resulting values state what % of values predicted as class x
        # had true label y.
        normalization_factor = 1/confusion_matrix_data.sum(axis=1)
        for i in range(confusion_matrix_data.shape[0]):
            for j in range(confusion_matrix_data.shape[2]):
                confusion_matrix_data[i, :, j] = confusion_matrix_data[i, :, j] * normalization_factor[i, j]
        # this plot will not incorporate std
        # also no plot for individual runs
        create_confusion_matrix_plot(confusion_matrix_data,
                                     out_base_path=Path.joinpath(plot_base_path, Path("Confusion_Matrix_Mean")),
                                     out_filename="Confusion_Matrix_Plot")
    if "distribution recall correlation" not in exclude_plot_types:
        print("Creating class distribution recall correlation plots.")
        distribution_recall_base_path = Path.joinpath(plot_base_path, Path("Class_Recall_Distribution"))
        if not Path(distribution_recall_base_path).is_dir():
            distribution_recall_base_path.mkdir()
        class_distribution_data = np.asarray(log_data["Class Distribution All"])
        # class recall can be calculated by dividing number of true positives of class
        # (entry in diagonal of confusion matrix) by the number of samples with true label(TP + FN) of said class
        # (row in confusion matrix)
        confusion_matrix_data = np.asarray(log_data["Confusion Matrix All"])
        # axis = 0 run , axis = 1 iteration, axis 2 = true labels, axis 3 = predicted labels
        summed_confusion_matrix_data = np.sum(confusion_matrix_data, axis=2)
        class_recall_data = np.zeros((confusion_matrix_data.shape[0], confusion_matrix_data.shape[1],
                                      len(CIFAR_CLASSES)))
        for run_idx in range(confusion_matrix_data.shape[0]):
            for iteration_idx in range(confusion_matrix_data.shape[1]):
                for class_idx in range(len(CIFAR_CLASSES)):
                    if summed_confusion_matrix_data[run_idx][iteration_idx][class_idx] != 0:
                        class_recall_data[run_idx][iteration_idx][class_idx] = \
                            confusion_matrix_data[run_idx][iteration_idx][class_idx][class_idx] / \
                            summed_confusion_matrix_data[run_idx][iteration_idx][class_idx]
        class_distribution_data = class_distribution_data.swapaxes(0, 1).swapaxes(1, 2)
        class_recall_data = class_recall_data.swapaxes(0, 1).swapaxes(1, 2)
        # calculate plot bounds
        y_bounds = calculate_bounds(class_distribution_data)
        x_bounds = calculate_bounds(class_recall_data)

        for iteration in range(class_recall_data.shape[0]):
            create_scatter_plot(class_recall_data[iteration], class_distribution_data[iteration],
                                out_path=Path.joinpath(distribution_recall_base_path,
                                                       Path(f"Class_Recall_Distribution_Plot_{iteration}.png")),
                                group_labels=CIFAR_CLASSES, colors=np.asarray(plt.get_cmap("tab10").colors),
                                show_legend=False, x_bounds=x_bounds, y_bounds=y_bounds,
                                title=f"Class Distribution - Class Recall Iteration {iteration}")

    if "distribution precision correlation" not in exclude_plot_types:
        print("Creating class distribution precision correlation plots.")
        distribution_precision_base_path = Path.joinpath(plot_base_path, Path("Class_Precision_Distribution"))
        if not Path(distribution_precision_base_path).is_dir():
            distribution_precision_base_path.mkdir()
        class_distribution_data = np.asarray(log_data["Class Distribution All"])
        # class precision can be calculated by dividing number of true positives of class
        # (entry in diagonal of confusion matrix) by the number of samples with true label(TP + FP) of said class
        # (column in confusion matrix)
        confusion_matrix_data = np.asarray(log_data["Confusion Matrix All"])
        # axis = 0 run , axis = 1 iteration, axis 2 = true labels, axis 3 = predicted labels
        summed_confusion_matrix_data = np.sum(confusion_matrix_data, axis=3)
        class_precision_data = np.zeros((confusion_matrix_data.shape[0], confusion_matrix_data.shape[1],
                                         len(CIFAR_CLASSES)))

        for run_idx in range(confusion_matrix_data.shape[0]):
            for iteration_idx in range(confusion_matrix_data.shape[1]):
                for class_idx in range(len(CIFAR_CLASSES)):
                    if summed_confusion_matrix_data[run_idx][iteration_idx][class_idx] != 0:
                        class_precision_data[run_idx][iteration_idx][class_idx] = \
                            confusion_matrix_data[run_idx][iteration_idx][class_idx][class_idx] / \
                            summed_confusion_matrix_data[run_idx][iteration_idx][class_idx]
        class_distribution_data = class_distribution_data.swapaxes(0, 1).swapaxes(1, 2)
        class_precision_data = class_precision_data.swapaxes(0, 1).swapaxes(1, 2)
        # calculate plot bounds
        y_bounds = calculate_bounds(class_distribution_data)
        x_bounds = calculate_bounds(class_precision_data)

        for iteration in range(class_precision_data.shape[0]):
            create_scatter_plot(class_precision_data[iteration], class_distribution_data[iteration],
                                out_path=Path.joinpath(distribution_precision_base_path,
                                                       Path(f"Class_Precision_Distribution_Plot_{iteration}.png")),
                                group_labels=CIFAR_CLASSES, colors=np.asarray(plt.get_cmap("tab10").colors),
                                show_legend=False, x_label="Precision", x_bounds=x_bounds, y_bounds=y_bounds,
                                title=f"Class Distribution - Class Precision Iteration {iteration}")

    if "distribution accuracy correlation" not in exclude_plot_types:
        print("Creating class distribution accuracy correlation plots.")
        distribution_accuracy_base_path = Path.joinpath(plot_base_path, Path("Class_Accuracy_Distribution"))
        if not Path(distribution_accuracy_base_path).is_dir():
            distribution_accuracy_base_path.mkdir()
        class_distribution_data = np.asarray(log_data["Class Distribution All"])
        # class accuracy can be calculated as TP + TN / TP+TN+FP+FN <-- 1
        # Or 1-row_of_confusion_matrix-column_of_confusion_matrix + 2*diagonal entry
        confusion_matrix_data = np.asarray(log_data["Confusion Matrix All"])
        # axis = 0 run , axis = 1 iteration, axis 2 = true labels, axis 3 = predicted labels
        summed_row_confusion_matrix_data = np.sum(confusion_matrix_data, axis=2)
        summed_column_confusion_matrix_data = np.sum(confusion_matrix_data, axis=3)
        class_accuracy_data = np.empty(
            (confusion_matrix_data.shape[0], confusion_matrix_data.shape[1], len(CIFAR_CLASSES)))
        for run_idx in range(confusion_matrix_data.shape[0]):
            for iteration_idx in range(confusion_matrix_data.shape[1]):
                for class_idx in range(len(CIFAR_CLASSES)):
                    class_accuracy_data[run_idx][iteration_idx][class_idx] =\
                        1 - summed_row_confusion_matrix_data[run_idx][iteration_idx][class_idx]\
                        - summed_column_confusion_matrix_data[run_idx][iteration_idx][class_idx] +\
                        (2*confusion_matrix_data[run_idx][iteration_idx][class_idx][class_idx])
        class_distribution_data = class_distribution_data.swapaxes(0, 1).swapaxes(1, 2)
        class_accuracy_data = class_accuracy_data.swapaxes(0, 1).swapaxes(1, 2)
        # calculate plot bounds
        y_bounds = calculate_bounds(class_distribution_data)
        x_bounds = calculate_bounds(class_accuracy_data)

        for iteration in range(class_accuracy_data.shape[0]):
            create_scatter_plot(class_accuracy_data[iteration], class_distribution_data[iteration],
                                out_path=Path.joinpath(distribution_accuracy_base_path,
                                                       Path(f"Class_Accuracy_Distribution_Plot_{iteration}.png")),
                                group_labels=CIFAR_CLASSES, colors=np.asarray(plt.get_cmap("tab10").colors),
                                show_legend=False, x_label="Accuracy", x_bounds=x_bounds, y_bounds=y_bounds,
                                title=f"Class Distribution - Class Accuracy Iteration {iteration}")
    if plot_individual_runs:
        # individual plots
        print("Creating individual plots..")
        individual_plot_path = Path.joinpath(plot_base_path, Path("individual_runs"))
        if not individual_plot_path.is_dir():
            individual_plot_path.mkdir()
        # accuracy
        if "accuracy" not in exclude_plot_types:
            print("Creating accuracy plots.")
            y_data = []
            x_data = []
            labels = []
            accuracy_data = log_data["Accuracy All"]
            for run_idx, run in enumerate(accuracy_data):
                y_data.append(np.asarray(run))
                x_data.append((np.arange(log_data["Iterations"]+1) * log_data["Budget"]) + log_data["Initial Split"])
                labels.append(f"Run {run_idx}")
            create_line_plot(x_data, y_data, out_path=Path.joinpath(individual_plot_path,
                                                                    Path("Accuracy Plot Individual Runs.png")),
                             labels=labels,
                             title="Accuracy Plot Individual Runs", show_legend=False)

        if "class distribution" not in exclude_plot_types:
            print("Creating class distribution plots.")
            # class distributions
            data = np.asarray(log_data["Class Distribution All"])
            create_class_dist_plot(data, Path.joinpath(individual_plot_path, Path("Class_Distribution")),
                                   "Class_Distribution_All_Plot")
            # class distribution information gain
        if "class distribution information gain" not in exclude_plot_types:
            print("Creating information gain plots.")
            y_data = []
            x_data = []
            labels = []
            information_gain_data = log_data["Information Gain All"]
            for run_idx, run in enumerate(information_gain_data):
                y_data.append(np.asarray(run))
                x_data.append((np.arange(log_data["Iterations"]+1) * log_data["Budget"]) + log_data["Initial Split"])
                labels.append(f"Run {run_idx}")
            create_line_plot(x_data, y_data, out_path=Path.joinpath(individual_plot_path,
                             Path("Information Gain Individual Runs.png")), labels=labels,
                             title="Information Gain Individual Runs", y_label="Information Gain", show_legend=False)


def create_scatter_plot(x_data, y_data, out_path, group_labels,
                        title="Class Distribution Recall Correlation", x_label="Recall",
                        y_label="Class Distribution", x_bounds=None, y_bounds=None, colors=None, show_legend=True):
    """
    Creates a 2D-scatter plot. Expects 2D-numpy array for x and y data. First dimension represents groups - groups will
    share color.
    :param x_data: 2D-numpy array of x-coordinates. First dimension refers to groups.
    :param y_data: 2D-numpy array of y-coordinates. First dimension refers to groups.
    :param out_path: desired out_file path. Expects pathlib filepath.
    :param group_labels: list of labels for each group.
    :param title: title of the created plot.
    :param x_label: label of x-axis.
    :param y_label: label of y-axis.
    :param x_bounds: tuples of minimum and maximum bound of x axis. If None - will be inferred of x_data.
    :param y_bounds: tuples of minimum and maximum bound of x axis. If None - will be inferred of y_data.
    :param colors: numpy array of color triples. Defines colors for groups.
    :param show_legend: toggle whether legend is drawn, default: False.
    :return:
    """
    if x_bounds is None:
        x_bounds = calculate_bounds(x_data)
    if y_bounds is None:
        y_bounds = calculate_bounds(y_data)
    fig, ax = plt.subplots(figsize=(7, 4))
    for group_idx in range(x_data.shape[0]):
        ax.scatter(x_data[group_idx], y_data[group_idx], alpha=0.8, c=np.expand_dims(colors[group_idx], 0), s=30,
                   label=group_labels[group_idx])
    if show_legend:
        ax.legend(loc='lower right')
    ax.set_ylim(y_bounds)
    ax.set_xlim(x_bounds)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def create_class_dist_plot(data, out_base_path, out_filename, labels=None, error=None, title="Class Distribution",
                           x_label="Classes", y_label="Class Distribution", show_legend=True):
    """
    Creates bar plots of class distribution.
    :param data: distribution data as 2 or 3 dimensional numpy array. [optional(runs), iteration, classes]
    :param out_base_path: base path of created directory - expects pathlib path.
    :param out_filename: desired out_file name as string, will have an iteration index appended to it.
    :param labels: class labels.
    :param error: error data as 2 or 3 dimensional numpy array. [optional(runs), iteration, classes]
    :param title: title of the plot, will have an iteration index appended to it.
    :param x_label: label of x-axis.
    :param y_label: label of y-axis.
    :param show_legend: toggle whether legend is drawn, default: False.
    :return:
    """
    if not Path(out_base_path).is_dir():
        out_base_path.mkdir()
    if len(data.shape) == 2:
        data = np.expand_dims(data, 0)
    y_bounds = calculate_bounds(data)
    if error is not None:
        y_bounds = (y_bounds[0] - np.max(error), y_bounds[1] + np.max(error))
        if len(error.shape) == 2:
            error = np.expand_dims(error, 0)
    if labels is None:
        labels = []
        for run_idx in range(data.shape[0]):
            labels.append(f"run:{run_idx}")
    if data.shape[0] > 10:
        np.random.seed(12)
        colors = np.random.rand(data.shape[0], 3)
    else:
        colors = np.asarray(plt.get_cmap("tab10").colors)
    for iteration_idx in range(data.shape[1]):
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(CIFAR_CLASSES))
        for run_idx in range(data.shape[0]):
            bar_width = 0.8 / data.shape[0]
            if error is not None:
                ax.bar(x + bar_width * run_idx - 0.4, data[run_idx, iteration_idx, :],
                       yerr=error[run_idx, iteration_idx, :], width=bar_width, color=colors[run_idx],
                       label=labels[run_idx], align="edge")
            else:
                ax.bar(x + bar_width * run_idx - 0.4, data[run_idx, iteration_idx, :], width=bar_width,
                       color=colors[run_idx], label=labels[run_idx],
                       align="edge")
        if show_legend:
            ax.legend(loc='lower right')
        ax.set_xticks(np.arange(len(CIFAR_CLASSES)))
        ax.set_xticklabels(CIFAR_CLASSES)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")
        ax.set_ylim(y_bounds)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{title} Iteration: {iteration_idx}")
        plt.savefig(Path.joinpath(out_base_path, Path(f"{out_filename}_{iteration_idx}.png")), dpi=200,
                    bbox_inches='tight')
        plt.close(fig)


def create_confusion_matrix_plot(data, out_base_path, out_filename, x_label="True Label", y_label="Predicted Label"):
    """
    Creates a confusion matrix plot for each iteration of a (averaged) run.
    :param data: Confusion Matrix Data: [iteration, true label, predicted label]
    :param out_base_path: base path of created directory - expects pathlib path.
    :param out_filename: desired out_file name as string, will have an iteration index appended to it.
    :param x_label: label of x-axis.
    :param y_label: label of y-axis.
    :return:
    """
    if not Path(out_base_path).is_dir():
        out_base_path.mkdir()
    for iteration_idx in range(data.shape[0]):
        fig, ax = plt.subplots(figsize=(7.2, 6))
        im, cbar = heatmap(data[iteration_idx], CIFAR_CLASSES, CIFAR_CLASSES, ax=ax,
                           cmap="YlGn", cbarlabel="Fraction Labelled", xlabel=x_label, ylabel=y_label)
        annotate_heatmap(im, valfmt="{x:.3f}")

        fig.tight_layout()
        plt.savefig(Path.joinpath(out_base_path, Path(f"{out_filename}_{iteration_idx}.png")), dpi=200,
                    bbox_inches='tight')
        plt.close(fig)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", xlabel="", ylabel="", **kwargs):
    """
    Taken from https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
    :param data:
    :param row_labels:
    :param col_labels:
    :param ax:
    :param cbar_kw:
    :param cbarlabel:
    :param xlabel:
    :param ylabel:
    :param kwargs:
    :return:
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    Taken from https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def calculate_bounds(data):
    """
    Helper function. Calculates plot bounds for one axis based on provided data and the PLOT_PADDING_FACTOR.
    :param data: Expects either np.array or a list of np.arrays.
    :return:
    """
    if type(data) == list:
        min_entry = min([np.min(np_array) for np_array in data])
        max_entry = max([np.max(np_array) for np_array in data])
    else:
        min_entry = np.min(data)
        max_entry = np.max(data)
    lower_bound = min_entry - ((max_entry - min_entry) / PLOT_PADDING_FACTOR)
    upper_bound = max_entry + ((max_entry - min_entry) / PLOT_PADDING_FACTOR)

    return lower_bound, upper_bound


def single_plot_all(plot_individual_runs=False, exclude_plot_types=None):
    """
    Utility function. Passes settings to create_single_setting_plots() for all available merged plot files.
    :param plot_individual_runs:
    :param exclude_plot_types:
    :return:
    """
    if Path(MERGED_LOGS_PATH).is_dir():
        log_files = list(Path(MERGED_LOGS_PATH).glob('*.json'))
        for log_file in log_files:
            create_single_setting_plots(log_file.parts[-1], plot_individual_runs=plot_individual_runs,
                                        exclude_plot_types=exclude_plot_types)
    else:
        merge_similar_runs()
        single_plot_all(plot_individual_runs=plot_individual_runs, exclude_plot_types=exclude_plot_types)


if __name__ == '__main__':
    fire.Fire({'merge': merge_similar_runs,
               'single_plot': create_single_setting_plots,
               'multi_plot': create_plots_over_setting,
               'single_plot_all': single_plot_all})
