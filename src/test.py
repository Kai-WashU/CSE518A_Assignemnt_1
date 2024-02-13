import parse
import aggregators
import evaluate
import unittest

class Tester(unittest.TestCase):
    # Sanity check that we parsed correctly. We expect 10 labels per task, 800 tasks
    def test_parse(self):
        p = parse.RTEParser()
        p.parse("rte.standardized.tsv")

        for task_id in p.data_by_task:
            workers = p.get_workers_for_task(task_id)
            self.assertTrue(len(workers) == 10,
                            f"{task_id} had {len(workers)} labels; expected 10.")
        num_tasks = len(list(p.data_by_task.keys()))
        self.assertTrue(num_tasks == 800,
                        f"Had {num_tasks} tasks; expected 800.")

    def test_extrapolation(self):
        p = parse.RTEParser()
        p.parse("rte.standardized.tsv")
        extrapolated = p.generate_extrapolated_dataset()

        self.assertTrue(len(extrapolated) == 164)
        for worker_id in extrapolated:
            self.assertTrue(len(extrapolated[worker_id]) == 800)
            for task_id in extrapolated[worker_id]:
                self.assertTrue(extrapolated[worker_id][task_id] == 1 or extrapolated[worker_id][task_id] == -1)
        
        print("Extrapolation:")
        subsample = p.generate_extrapolated_subsample(20)
        good = aggregators.SVDAggregator.find_good_worker(extrapolated, p.data_by_task, subsample)
        aggregators.SVDAggregator.aggregate(extrapolated, subsample, good)
        print("=========")
    
    def test_perfect_svd(self):
        p = parse.RTEParser()
        p.parse("rte.standardized.tsv")
        perfect = {} # This will contain the actual labels for every worker

        for worker_id in p.data_by_worker:
            perfect[worker_id] = {}
            for task_id in p.data_by_task:
                perfect[worker_id][task_id] = p.data_by_task[task_id].true_label
        
        answer = aggregators.SVDAggregator.aggregate(perfect, p.generate_extrapolated_subsample(164), "A19IBSKBTABMR3")
        e = evaluate.Evaluator()
        error = e.evaluate(answer)

        self.assertTrue(error == 0, f"Error should be 0, was {error}")
        

if __name__ == '__main__':
    unittest.main()