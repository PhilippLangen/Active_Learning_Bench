import json
from pathlib import Path

import fire
import numpy as np

from active_learning_benchmark import ActiveLearningBench

TOTAL_BUDGET = 40000


def add_new_jobs(json_queue, strategies=["random_sampling"], initial_training_splits=[1000], batch_sizes=[32],
                 target_layers=[4], budgets=[1000], model="simplecnn", runs=5):
    job_list = []
    json_queue_path = Path(json_queue)
    if json_queue_path.is_file():
        with json_queue_path.open() as json_file:
            job_list = json.load(json_file)
            json_file.close()
    for strategy in strategies:
        for initial_training_split in initial_training_splits:
            for batch_size in batch_sizes:
                for target_layer in target_layers:
                    for budget in budgets:
                        for run in range(runs):
                            new_job = {"Strategy": strategy, "Budget": budget, "Initial Split": initial_training_split,
                                       "Iterations": int((TOTAL_BUDGET-initial_training_split)/budget),
                                       "Batch Size": batch_size, "Target Layer": target_layer, "Model": model}
                            job_list.append(new_job)
    with json_queue_path.open('w', encoding='utf-8') as file:
        json.dump(job_list, file, ensure_ascii=False)
        file.close()


def run_queue(queue_json):
    queue_file = Path(queue_json)
    if queue_file.is_file():
        with queue_file.open() as file:
            queue = json.load(file)
            file.close()
            while len(queue) > 0:
                job = queue.pop(0)
                log_file = f"{job['Strategy']}_{job['Budget']}_{job['Initial Split']}_{job['Iterations']}_" \
                    f"{job['Batch Size']}_{job['Target Layer']}_{job['Model']}"
                ActiveLearningBench(labeling_strategy=job['Strategy'], logfile=log_file,
                                    initial_training_size=job['Initial Split'], batch_size=job['Batch Size'],
                                    budget=job['Budget'], iterations=job['Iterations'],
                                    target_layer=job['Target Layer'], model=job['Model']).run()
                with queue_file.open('w', encoding='utf-8') as modified_file:
                    json.dump(queue, modified_file, ensure_ascii=False)
                    file.close()
            print(f"Job queue {queue_file} has been completed!")
            queue_file.unlink()
    else:
        print(f"{queue_json} not found!")
        raise SystemExit


def split_workload(json_queue, num_splits, remove_original_file=True):
    queue_file = Path(json_queue)
    if queue_file.is_file():
        with queue_file.open() as file:
            workload = np.asarray(json.load(file))
            file.close()
    else:
        print(f"{json_queue} not found!")
        raise SystemExit
    work_splits = np.array_split(workload, num_splits)
    for work_split_index, work_split in enumerate(work_splits):
        work_split_list = work_split.tolist()
        work_split_filename = json_queue[:-5]
        work_split_filename = f"{work_split_filename}_part_{work_split_index}.json"
        with Path(work_split_filename).open('w', encoding='utf-8') as split_file:
            json.dump(work_split_list, split_file)
            split_file.close()
    if remove_original_file:
        queue_file.unlink()


if __name__ == '__main__':
    fire.Fire({'add_new_jobs': add_new_jobs,
               'run_queue': run_queue,
               'split_workload': split_workload
               })
