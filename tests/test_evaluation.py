import unittest

import numpy as np

from retail_retention.evaluation import (
    classification_metrics,
    expected_calibration_error,
    total_error_cost,
)


class EvaluationTests(unittest.TestCase):
    def test_classification_metrics(self):
        result = classification_metrics(
            [False, False, True, True],
            [0.1, 0.7, 0.4, 0.9],
            threshold=0.5,
        )
        self.assertEqual(result["FalsePositives"], 1)
        self.assertEqual(result["FalseNegatives"], 1)
        self.assertEqual(result["TruePositives"], 1)
        self.assertEqual(result["TrueNegatives"], 1)

    def test_total_error_cost(self):
        self.assertEqual(
            total_error_cost(3, 2, fp_cost=4, fn_cost=10),
            32,
        )

    def test_perfect_calibration(self):
        result = expected_calibration_error(
            np.array([False, True]),
            np.array([0.0, 1.0]),
        )
        self.assertEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()

