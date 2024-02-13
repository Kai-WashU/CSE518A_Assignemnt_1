import parse
import aggregators

REPITITIONS = 10
METHODS: dict[str, aggregators.Aggregator] = {
    "Majority Vote": aggregators.MajorityVoteAggregator,
    "Estimation Maximization": aggregators.EstimationMaximizationAggregator,
    "SVD": aggregators.SVDAggregator
}

class Evaluator:
    def __init__(self) -> None:
        self.parser = parse.RTEParser()
        self.parser.parse("rte.standardized.tsv")

        # Task ID --> Answer
        self.answers: dict[str, int] = {}
        for task_id in self.parser.data_by_task:
            self.answers[task_id] = self.parser.data_by_task[task_id].true_label

        # Method --> k (size) --> trial iteration errors
        self.errors: dict[str, dict[int, list[float]]] = {}
        # Method --> k (size) --> average error
        self.average_errors: dict[str, dict[int, float]] = {}

        for method in METHODS:
            self.errors[method]: dict[int, list[float]] = {}
            self.average_errors[method]: dict[int, float] = {}

    # Return the error (incorrect / total) for a certain determination
    def evaluate(self, aggregations: dict[str, int]) -> float:
        incorrect = 0.0

        for task_id in aggregations:
            if aggregations[task_id] == 0:
                incorrect += 0.5
            elif self.answers[task_id] != aggregations[task_id]:
                incorrect += 1
        
        return incorrect / len(aggregations)

    def run_trials(self, size: int) -> None:
        # Run REPITITIONS # of trials
        for _ in range(REPITITIONS):
            subsample = self.parser.generate_subsample(size)

            # Run this subsample on every method
            for method in METHODS:
                if method == "SVD":
                    good_worker = aggregators.SVDAggregator.find_good_worker(self.parser.data_by_worker, self.parser.data_by_task, subsample)
                    aggregations = aggregators.SVDAggregator.aggregate(self.parser.data_by_worker, subsample, good_worker)
                else:
                    aggregations = METHODS[method].aggregate(self.parser.data_by_worker, subsample)
                error = self.evaluate(aggregations)
                if size not in self.errors[method]:
                    self.errors[method][size] = []
                self.errors[method][size].append(error)
        
        # Calculate the average errors for each method
        for method in METHODS:
            average = 0
            for error in self.errors[method][size]:
                average += error
            self.average_errors[method][size] = average / REPITITIONS
    
    def main(self) -> None:
        for size in range(1, 11):
            self.run_trials(size)
        print("Individual Trial Errors:")
        print(self.errors)
        print("Average Errors:")
        print(self.average_errors)

        # Perform SVD on extrapolated samples
        extrapolated_trial_errors: dict[int, list[float]] = {}
        extrapolated_dataset: dict[str, dict[str, int]] = self.parser.generate_extrapolated_dataset()
        extrapolated_sizes = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 35, 40, 50, 60, 80, 100]
        for size in extrapolated_sizes:
            extrapolated_trial_errors[size] = []

            for _ in range(REPITITIONS):
                subsample = self.parser.generate_extrapolated_subsample(size)
                good_worker = aggregators.SVDAggregator.find_good_worker(extrapolated_dataset, self.parser.data_by_task, subsample)
                aggregations = aggregators.SVDAggregator.aggregate(extrapolated_dataset, subsample, good_worker)
                extrapolated_trial_errors[size].append(self.evaluate(aggregations))
        
        average_extrapolated_errors: dict[int, float] = {}
        for size in extrapolated_trial_errors:
            total_error = 0
            for error in extrapolated_trial_errors[size]:
                total_error += error
            average_extrapolated_errors[size] = total_error / REPITITIONS
        
        print("Extraploated Trial Errors:")
        print(extrapolated_trial_errors)
        print("Extrapolated SVD Errors:")
        print(average_extrapolated_errors)

if __name__ == "__main__":
    e = Evaluator()
    e.main()