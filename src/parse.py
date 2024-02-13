import dataclasses
import random

# Parse .tsv into python object
class TableParser:
    def __init__(self):
        self.columns: list[str] = []
        self.data: list[list] = []
    
    def parse_tsv(self, filepath: str):
        with open(filepath) as file:
            # Read the column names
            self.columns = file.readline().strip().split('\t')

            # Read the values into the table
            for line in file:
                values = line.strip().split('\t')
                self.data.append(values)

@dataclasses.dataclass
class TaskEntry:
    true_label: int
    labels: dict[str, int]

class RTEParser(TableParser):
    def __init__(self):
        super().__init__()
        
        # We don't need `!amt_annotation_ids`, and convert '0' labels to '-1'
        # Need to be able to access by `!amt_worker_ids` and `orig_id`
        # Renaming columns for convenience

        # Worker ID --> Task ID --> Given Label
        self.data_by_worker: dict[str, dict[str, int]] = {}
        # Task ID --> (True Label, Worker ID --> Given Label)
        self.data_by_task: dict[str, TaskEntry] = {}

    # Apply filtering on data specific to Assignment 1
    def parse(self, filepath: str):
        super().parse_tsv(filepath)

        worker_id_index = self.columns.index("!amt_worker_ids")
        task_id_index = self.columns.index("orig_id")
        given_label_index = self.columns.index("response")
        true_label_index = self.columns.index("gold")
        
        for entry in self.data:
            worker_id = entry[worker_id_index]
            task_id = entry[task_id_index]
            given_label = int(entry[given_label_index])
            true_label = int(entry[true_label_index])

            # Convert 0's to -1's
            if given_label == 0:
                given_label = -1
            if true_label == 0:
                true_label = -1

            # Place data into structures
            if worker_id not in self.data_by_worker:
                self.data_by_worker[worker_id] = {}
            self.data_by_worker[worker_id][task_id] = given_label

            if task_id not in self.data_by_task:
                self.data_by_task[task_id] = TaskEntry(true_label, {})
            self.data_by_task[task_id].labels[worker_id] = given_label
    
    def get_workers_for_task(self, task_id: str) -> list[str]:
        return list(self.data_by_task[task_id].labels.keys())

    # Dictionary of Task ID --> {Worker ID,...}
    def generate_subsample(self, size: int) -> dict[str, set[str]]:
        assert(size <= 10)
        subsample: dict[str, set[str]] = {}

        # For each task
        for task_id in self.data_by_task:
            workers = self.get_workers_for_task(task_id)
            chosen_workers: set[str] = set()

            # Choose some number of workers randomly
            while len(chosen_workers) != size:
                index = random.randint(0, len(workers) - 1)
                chosen_workers.add(workers[index])
                del workers[index]

            subsample[task_id] = chosen_workers
            assert(len(chosen_workers) == size)

        return subsample
    
    def generate_extrapolated_dataset(self) -> dict[str, dict[str, int]]:
        # Find out empirical accuracies of workers
        accuracies: dict[str, float] = {}
        for worker_id in self.data_by_worker:
            correct = 0
            total = 0
            for task_id in self.data_by_worker[worker_id]:
                total += 1
                if self.data_by_worker[worker_id][task_id] == self.data_by_task[task_id].true_label:
                    correct += 1
            accuracies[worker_id] = float(correct) / total
        
        # Fill in all values
        extended_data_by_worker: dict[str, dict[str, int]] = {}
        for worker_id in self.data_by_worker:
            extended_data_by_worker[worker_id] = {}
            remaining_tasks = set(self.data_by_task.keys())

            # Copy existing data
            for task_id in self.data_by_worker[worker_id]:
                extended_data_by_worker[worker_id][task_id] = self.data_by_worker[worker_id][task_id]
                remaining_tasks.remove(task_id)
            
            # Choose correct synthetic tasks
            remaining_tasks = list(remaining_tasks)
            tasks_to_choose = int(len(remaining_tasks) * accuracies[worker_id])
            while tasks_to_choose > 0:
                choice = random.randint(0, len(remaining_tasks) - 1)
                extended_data_by_worker[worker_id][remaining_tasks[choice]] = self.data_by_task[remaining_tasks[choice]].true_label
                del remaining_tasks[choice]
                tasks_to_choose -= 1
            
            # Fill in incorrect synthetic tasks
            for task_id in remaining_tasks:
                extended_data_by_worker[worker_id][task_id] = -self.data_by_task[task_id].true_label
        
        return extended_data_by_worker
    
    def generate_extrapolated_subsample(self, size: int) -> dict[str, set[str]]:
        subsample: dict[str, set[str]] = {}

        for task_id in self.data_by_task:
            workers = list(self.data_by_worker.keys())
            chosen_workers: set[str] = set()

            # Choose some number of workers randomly
            while len(chosen_workers) != size:
                index = random.randint(0, len(workers) - 1)
                chosen_workers.add(workers[index])
                del workers[index]

            subsample[task_id] = chosen_workers
            assert(len(chosen_workers) == size)
        
        return subsample

if __name__ == "__main__":
    p = RTEParser()
    p.parse("rte.standardized.tsv")
    print(p.generate_subsample(5))