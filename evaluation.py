import json
from collections import defaultdict
from pathlib import Path

import numpy as np


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
        with open(log) as json_file:
            json_data = json.load(json_file)
            key = (json_data["Strategy"], json_data["Budget"], json_data["Initial Split"],
                   json_data["Epochs"], json_data["Iterations"], json_data["Batch Size"])
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
                       "Iterations": key[4], "Batch Size": key[5], "Accuracy All": acc, "Accuracy Mean": acc_mean,
                       "Accuracy Std": acc_std, "Class Distribution All": class_dist,
                       "Class Distribution Mean": class_dist_mean, "Class Distribution Std": class_dist_std,
                       "Confusion Matrix All": conf_mat, "Confusion Matrix Mean": conf_mat_mean,
                       "Confusion Matrix Std": conf_mat_std}
        # generate a filename by settings
        target_file = Path(f"{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}_{key[5]}.json")
        # create json file
        with Path.joinpath(Path(target_path_base, target_file)).open('w', encoding='utf-8') as file:
            json.dump(merged_dict, file, ensure_ascii=False)










if __name__ == '__main__':
    merge_similar_runs()