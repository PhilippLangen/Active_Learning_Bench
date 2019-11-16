import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PLOT_PADDING_FACTOR = 40
CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def merge_similar_runs():
    """
    Combine results of runs with identical settings. This function will automatically group log files by their settings
    and creates a merged log for each unique setting combination.
    :return:
    """
    # create/ensure existence of log directories
    target_path_base = "./logs/merged_logs"
    source_path_base = "./logs"
    if not Path(source_path_base).is_dir():
        print(f"Logfile directory {source_path_base} not found! Run experiments before using the evaluation tool")
        exit(1)
    if not Path(target_path_base).is_dir():
        Path(target_path_base).mkdir()
    source_path_base = Path(source_path_base)
    # find all json files in ./logs
    log_files = list(source_path_base.glob('*.json'))
    log_map = defaultdict(lambda: list())
    # create list of logs for each unique settings key
    for idx, log in enumerate(log_files):
        with log.open() as json_file:
            json_data = json.load(json_file)
            key = (json_data["Strategy"], json_data["Budget"], json_data["Initial Split"],
                   json_data["Epochs"], json_data["Iterations"], json_data["Batch Size"],
                   json_data["Learning Rate"], json_data["Target Layer"])
            log_map[key].append(json_data)
    # loop over each setting key
    for key, log_list in log_map.items():
        # initialize data containers
        acc = np.empty(shape=(0, key[4]+1))
        class_dist = np.empty(shape=(0, key[4]+1, 10))
        conf_mat = np.empty(shape=(0, key[4]+1, 10, 10))
        # accumulate date over multiple runs
        for log in log_list:
            acc = np.append(acc, np.expand_dims(np.asarray(log["Accuracy"]), axis=0), axis=0)
            class_dist = np.append(class_dist, np.expand_dims(np.asarray(log["Class Distribution"]), axis=0), axis=0)
            conf_mat = np.append(conf_mat, np.expand_dims(np.asarray(log["Confusion Matrix"]), axis=0), axis=0)
        # create mean and standard deviation info over runs
        acc_mean = np.mean(acc, axis=0)
        class_dist_mean = np.mean(class_dist, axis=0)
        acc_std = np.std(acc, axis=0)
        class_dist_std = np.std(class_dist, axis=0)
        conf_mat_mean = np.mean(conf_mat, axis=0)
        conf_mat_std = np.std(conf_mat, axis=0)
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
        # create json structure
        merged_dict = {"Strategy": key[0], "Budget": key[1], "Initial Split": key[2], "Epochs": key[3],
                       "Iterations": key[4], "Batch Size": key[5], "Learning Rate": key[6], "Target Layer": key[7],
                       "Accuracy All": acc, "Accuracy Mean": acc_mean, "Accuracy Std": acc_std,
                       "Class Distribution All": class_dist, "Class Distribution Mean": class_dist_mean,
                       "Class Distribution Std": class_dist_std, "Confusion Matrix All": conf_mat,
                       "Confusion Matrix Mean": conf_mat_mean, "Confusion Matrix Std": conf_mat_std}
        # generate a filename by settings
        target_file = Path(f"{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}_{key[5]}_{key[6]}_{key[7]}.json")
        # create json file
        with Path.joinpath(Path(target_path_base, target_file)).open('w', encoding='utf-8') as file:
            json.dump(merged_dict, file, ensure_ascii=False)


def create_single_setting_plots(merged_logfile, plot_individual_runs=True, exclude_plot_types=None):
    if exclude_plot_types is None:
        exclude_plot_types = {}
    # get log data
    logfile = Path(f"./logs/merged_logs/{merged_logfile}")
    if not logfile.is_file():
        print(f"invalid filename: {logfile} does not exist!")
    with logfile.open() as json_file:
        log_data = json.load(json_file)
    plot_base_path = Path(f"./plots/single_setting_plots/{merged_logfile[:-5]}")
    # create directory for all plots of logfile
    if not plot_base_path.is_dir():
        plot_base_path.mkdir(parents=True)
    exclude_plot_types = {x.lower() for x in exclude_plot_types}
    # mean plots
    # accuracy
    if 'accuracy' not in exclude_plot_types:
        y_data = np.asarray(log_data["Accuracy Mean"])
        error = np.asarray(log_data["Accuracy Std"])
        x_data = (np.arange(y_data.shape[0])*log_data["Budget"]) + log_data["Initial Split"]
        create_line_plot(x_data, y_data, Path.joinpath(plot_base_path, Path("Mean_Accuracy_Plot.png")), error=error,
                         labels=["mean_acc"])
    # class distribution
    if "class distribution" not in exclude_plot_types:
        mean_data = np.asarray(log_data["Class Distribution Mean"])
        error = np.asarray(log_data["Class Distribution Std"])
        create_class_dist_plot(mean_data, Path.joinpath(plot_base_path, Path("Class_Distribution_Mean")),
                               "Mean_Class_Distribution_Plot", error=error, title="Mean Class Distribution",
                               y_label="Mean Class Distribution")
    # TODO
    # class distribution entropy
    # confusion matrix

    # individual plots
    if plot_individual_runs:
        individual_plot_path = Path.joinpath(plot_base_path, Path("individual_runs"))
        # accuracy
        if "accuracy" not in exclude_plot_types:
            ind_acc_path = Path.joinpath(individual_plot_path, Path("Accuracy"))
            if not ind_acc_path.is_dir():
                ind_acc_path.mkdir(parents=True)
            y_data = np.asarray((log_data["Accuracy All"]))
            # calculate y_bounds over all iterations so they match throughout the plots
            y_bounds = (np.min(y_data) - ((np.max(y_data) - np.min(y_data)) / PLOT_PADDING_FACTOR),
                        np.max(y_data) + ((np.max(y_data) - np.min(y_data)) / PLOT_PADDING_FACTOR))
            x_data = (np.arange(y_data.shape[1]) * log_data["Budget"]) + log_data["Initial Split"]
            for i in range(y_data.shape[0]):
                create_line_plot(x_data, y_data[i], Path.joinpath(ind_acc_path, Path(f"Accuracy_Plot_Run_{i}.png")),
                                 ["acc"], y_bounds=y_bounds, title=f"Accuracy Plot Run {i}")
        if "class distribution" not in exclude_plot_types:
            # class distributions
            data = np.asarray(log_data["Class Distribution All"])
            create_class_dist_plot(data, Path.joinpath(individual_plot_path, Path("Class_Distribution")),
                                   "Class_Distribution_All_Plot")
            # TODO
            # class distribution entropy
            # confusion matrix


def create_line_plot(x_data, y_data, out_path, labels=None, error=None, x_bounds=None, y_bounds=None,
                     title="Accuracy Plot", x_label="Labelled Samples", y_label="Accuracy"):
    if len(y_data.shape) == 1:
        y_data = np.expand_dims(y_data, 0)
    if x_bounds is None:
        x_bounds = (np.min(x_data)-((np.max(x_data)-np.min(x_data))/PLOT_PADDING_FACTOR),
                    np.max(x_data)+((np.max(x_data)-np.min(x_data))/PLOT_PADDING_FACTOR))
    if y_bounds is None:
        y_bounds = (np.min(y_data) - ((np.max(y_data)-np.min(y_data))/PLOT_PADDING_FACTOR),
                    np.max(y_data) + ((np.max(y_data)-np.min(y_data))/PLOT_PADDING_FACTOR))
        if error is not None:
            y_bounds = (y_bounds[0] - np.max(error), y_bounds[1] + np.max(error))
            if len(error.shape) == 1:
                error = np.expand_dims(error, 0)
    if labels is None:
        labels = list()
        for i in range(y_data.shape[0]):
            labels.append("")
    if error is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in range(y_data.shape[0]):
            ax.errorbar(x=x_data, y=y_data[i], yerr=error[i], marker='o', markersize=4, label=labels[i])
        ax.legend(loc='lower right')
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in range(y_data.shape[0]):
            ax.plot(x_data, y_data[i], marker='o', markersize=4, label=labels[i])
        ax.legend(loc='lower right')
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)


def create_class_dist_plot(data, out_base_path, out_filename, labels=None, error=None, title="Class Distribution",
                           x_label="Classes", y_label="Class Distribution"):
    if not Path(out_base_path).is_dir():
        out_base_path.mkdir()

    if len(data.shape) == 2:
        data = np.expand_dims(data, 0)
    y_bounds = (np.min(data) - ((np.max(data) - np.min(data)) / PLOT_PADDING_FACTOR),
                np.max(data) + ((np.max(data) - np.min(data)) / PLOT_PADDING_FACTOR))
    if error is not None:
        y_bounds = (y_bounds[0] - np.max(error), y_bounds[1] + np.max(error))
        if len(error.shape) == 2:
            error = np.expand_dims(error, 0)
    if labels is None:
        labels = list()
        for i in range(data.shape[0]):
            labels.append(str(i))
    np.random.seed(12)
    colors = np.random.rand(data.shape[0], 3)
    if error is not None:
        for i in range(data.shape[1]):
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(len(CIFAR_CLASSES))
            for j in range(data.shape[0]):
                bar_width = 0.8 / data.shape[0]
                ax.bar(x + bar_width * j - 0.4, data[j, i, :], yerr=error[j, i, :], width=bar_width, color=colors[j],
                       label=labels[j], align="edge")
            ax.legend(loc='lower right')
            ax.set_xticklabels(CIFAR_CLASSES)
            ax.set_ylim(y_bounds)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f"{title} Iteration: {i}")
            plt.savefig(Path.joinpath(out_base_path, Path(f"{out_filename}_{i}.png")), dpi=200, bbox_inches='tight')
            plt.close(fig)
    else:
        for i in range(data.shape[1]):
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(len(CIFAR_CLASSES))
            for j in range(data.shape[0]):
                bar_width = 0.8/data.shape[0]
                ax.bar(x + bar_width*j-0.4, data[j, i, :], width=bar_width, color=colors[j], label=labels[j],
                       align="edge")
            ax.legend(loc='lower right')
            ax.set_xticklabels(CIFAR_CLASSES)
            ax.set_ylim(y_bounds)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f"{title} Iteration: {i}")
            plt.savefig(Path.joinpath(out_base_path, Path(f"{out_filename}_{i}.png")), dpi=200, bbox_inches='tight')
            plt.close(fig)


if __name__ == '__main__':
    merge_similar_runs()
    create_single_setting_plots("greedy_k_center_1000_1000_2_40_32_0.001_2.json",
                                exclude_plot_types={"AccuraCy", "Class Distribution"})
