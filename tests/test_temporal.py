import unittest

import pandas as pd

from retail_retention.models import purged_temporal_split


class TemporalSplitTests(unittest.TestCase):
    def test_label_overlap_is_purged(self):
        data = pd.DataFrame({
            "ReferenceDate": pd.to_datetime([
                "2020-01-01", "2020-02-01", "2020-03-01"
            ]),
            "LabelEndDate": pd.to_datetime([
                "2020-01-31", "2020-03-02", "2020-03-31"
            ]),
        })
        train, validation = purged_temporal_split(
            data,
            train_start=pd.Timestamp("2020-01-01"),
            validation_start=pd.Timestamp("2020-03-01"),
            validation_end=pd.Timestamp("2020-03-01"),
        )
        self.assertEqual(train.index.tolist(), [0])
        self.assertEqual(validation.index.tolist(), [2])


if __name__ == "__main__":
    unittest.main()
