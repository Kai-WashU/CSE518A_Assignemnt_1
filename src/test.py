import parse
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

if __name__ == '__main__':
    unittest.main()