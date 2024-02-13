import abc
import random
import numpy
import parse
import sys

MAX_ITERATIONS = 20

class Aggregator(abc.ABC):
    # Returns dictionary of Task ID --> Aggregation
    @abc.abstractmethod
    def aggregate(data_by_worker: dict[str, dict[str, int]],
                  subsample: dict[str, set[str]]) -> dict[str, int]:
        pass

class MajorityVoteAggregator(Aggregator):
    def aggregate(data_by_worker: dict[str, dict[str, int]],
                  subsample: dict[str, set[str]]) -> dict[str, int]:
        aggregations: dict[str, int] = {}

        for task_id in subsample:
            workers = subsample[task_id]
            aggregate = 0

            for worker in workers:
                aggregate += data_by_worker[worker][task_id]
            
            if aggregate > 0:
                aggregations[task_id] = 1
            elif aggregate < 0:
                aggregations[task_id] = -1
            else:
                # In our evaluation step, we were instructed in class to
                # give aggregations of 0 an error of 0.5
                aggregations[task_id] = 0
        
        return aggregations

class EstimationMaximizationAggregator(Aggregator):
    def aggregate(data_by_worker: dict[str, dict[str, int]],
                  subsample: dict[str, set[str]]) -> dict[str, int]:
        # Dictionary from Worker ID --> weight
        worker_weights: dict[str, float] = {}
        for worker_id in data_by_worker:
            worker_weights[worker_id] = 1
        # Dictionary from Task ID --> aggregation
        old_estimate: dict[str, float] = {}
        for task_id in subsample:
            old_estimate[task_id] = 0

        iterations = 0
        while(iterations < MAX_ITERATIONS):
            estimate = EstimationMaximizationAggregator.weighted_majority(data_by_worker, subsample, worker_weights)
            if EstimationMaximizationAggregator.has_converged(old_estimate, estimate):
                return estimate
            else:
                old_estimate = estimate
                worker_weights = EstimationMaximizationAggregator.update_weights(data_by_worker, subsample, estimate)
                iterations += 1
        
        return old_estimate
    
    def has_converged(old_estimate: dict[str, int], new_estimate: dict[str, int]) -> bool:
        converged = True
        for task_id in new_estimate:
            if new_estimate[task_id] != old_estimate[task_id]:
                converged = False
        return converged

    def update_weights(data_by_worker: dict[str, dict[str, int]],
                       subsample: dict[str, set[str]],
                       estimate: dict[str, int]) -> dict[str, float]:
        # Dictionary from Worker ID --> # Correct
        worker_correct: dict[str, int] = {}
        # Dictonary from Worker ID --> # Tasks
        worker_total: dict[str, int] = {}
        # Dictionary from Worker ID --> Proportion correct
        new_weights: dict[str, float] = {}
        for task_id in subsample:
            for worker_id in subsample[task_id]:
                if worker_id not in worker_total:
                    worker_total[worker_id] = 0
                if worker_id not in worker_correct:
                    worker_correct[worker_id] = 0
                if worker_id not in new_weights:
                    new_weights[worker_id] = 0
                
                worker_total[worker_id] += 1
                if data_by_worker[worker_id][task_id] == estimate[task_id]:
                    worker_correct[worker_id] += 1
        
        for worker_id in new_weights:
            new_weights[worker_id] = 2 * (float(worker_correct[worker_id]) / worker_total[worker_id]) - 1
        
        return new_weights

    def weighted_majority(data_by_worker: dict[str, dict[str, int]],
                          subsample: dict[str, set[str]],
                          weights: dict[str, float]) -> dict[str, int]:
        aggregations: dict[str, int] = {}

        for task_id in subsample:
            workers = subsample[task_id]
            aggregate = 0

            for worker in workers:
                aggregate += data_by_worker[worker][task_id] * weights[worker]
            
            if aggregate > 0:
                aggregations[task_id] = 1
            elif aggregate < 0:
                aggregations[task_id] = -1
            # Actually manifest random label
            elif random.uniform(0, 1) < 0.5:
                aggregations[task_id] = -1
            else:
                aggregations[task_id] = 1
        
        return aggregations

class SVDAggregator(Aggregator):
    def find_good_worker(data_by_worker: dict[str, dict[str, int]],
                         data_by_task: dict[str, parse.TaskEntry],
                         subsample: dict[str, set[str]]) -> str:
        worker_correct: dict[str, int] = {}
        worker_total: dict[str, int] = {}

        for task_id in subsample:
            for worker_id in subsample[task_id]:
                if worker_id not in worker_correct:
                    worker_correct[worker_id] = 0
                if worker_id not in worker_total:
                    worker_total[worker_id] = 0
                if data_by_worker[worker_id][task_id] == data_by_task[task_id].true_label:
                    worker_correct[worker_id] += 1
                worker_total[worker_id] += 1
        
        for worker_id in worker_correct:
            if worker_correct[worker_id] * 2 > worker_total[worker_id]:
                return worker_id

    def aggregate(data_by_worker: dict[str, dict[str, int]],
                  subsample: dict[str, set[str]],
                  good_worker: str) -> dict[str, int]:
        # Calculate the top eigenvector
        (_, tasks, matrix) = SVDAggregator.convert_to_matrix(data_by_worker, subsample)
        symmetrical_matrix = numpy.matmul(matrix, numpy.transpose(matrix))
        (eigenvalues, eigenvectors) = numpy.linalg.eig(symmetrical_matrix)

        # Find the maximal eigenvalue
        max_index = 0
        for index in range(len(eigenvalues)):
            if eigenvalues[index] > eigenvalues[max_index]:
                max_index = index

        estimates: list[int] = []
        numpy.set_printoptions(threshold=20)
        for task in range(len(tasks)):
            value = eigenvectors[task][max_index]
            if value > 0:
                estimates.append(1)
            elif value < 0:
                estimates.append(-1)
            elif random.uniform(1, 0) < 0.5:
                estimates.append(-1)
            else:
                estimates.append(1)

        # Compare against the good worker
        aggregate = 0
        for task_id in data_by_worker[good_worker]:
            aggregate += data_by_worker[good_worker][task_id] * estimates[tasks[task_id]]
        
        # Return properly signed estimate
        result: dict[str, int] = {}
        for task in tasks:
            if aggregate >= 0:
                result[task] = estimates[tasks[task]]
            else:
                result[task] = -estimates[tasks[task]]
        
        return result

    def convert_to_matrix(data_by_worker: dict[str, dict[str, int]],
                          subsample: dict[str, set[str]]) -> tuple[dict[str, int], dict[str, int], numpy.matrix]:
        # Create mapping between rows and workers as well as columns and tasks
        tasks: dict[str, int] = {}
        workers: dict[str, int] = {}
        w_index = 0
        t_index = 0
        for task in subsample:
            if task not in tasks:
                tasks[task] = t_index
                t_index += 1
            for worker in subsample[task]:
                if worker not in workers:
                    workers[worker] = w_index
                    w_index += 1

        # Put the values into the matrix
        matrix = numpy.zeros((len(tasks), len(workers)))
        for task in tasks:
            for worker in subsample[task]:
                matrix.itemset((tasks[task], workers[worker]), data_by_worker[worker][task])
                
        return (workers, tasks, matrix)