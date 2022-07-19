import os
import unittest

import pandas as pd

from seclea_ai import SecleaAI


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(base_dir, "")
print(folder_path)


class TestSecleaAI(unittest.TestCase):

    # def test_check_features_different_names(self):
    #     parent_metadata = {
    #         "features": ["months_as_customer", "age", "policy_number", "policy_bind_date"]
    #     }
    #     test_dataset = pd.DataFrame([[3, 34, 339203, 34], [7, 32, 339103, 3]], columns=[1, 2, 3, 4])
    #     metadata = {}
    #     metadata = SecleaAI._check_features(dataset=test_dataset, metadata=metadata, parent_metadata=parent_metadata)
    #     print(metadata)
    #
    # def test_check_features_different_names_different_len(self):
    #     parent_metadata = {
    #         "features": ["months_as_customer", "age", "policy_number", "policy_bind_date"]
    #     }
    #     test_dataset = pd.DataFrame([[3, 34, 339203, 34], [7, 32, 339103, 3]], columns=[1, 2, 3, 4])
    #     metadata = {}
    #     metadata = SecleaAI._check_features(dataset=test_dataset, metadata=metadata, parent_metadata=parent_metadata)
    #     print(metadata)

    def test_get_dataset_type(self):
        # ARRANGE
        non_timeseries_dataset = pd.DataFrame(
            [[3, 34, 339203, 34], [7, 32, 339103, 3]], columns=[1, 2, 3, 4], index=[1, 2]
        )
        timeseries_dataset = pd.DataFrame(
            [[3, 34, 339203, 34], [7, 32, 339103, 3]],
            columns=[1, 2, 3, 4],
            index=["2022-01-01", "2022-01-02"],
        )

        # ACT
        res1 = SecleaAI._get_dataset_type(non_timeseries_dataset)
        res2 = SecleaAI._get_dataset_type(timeseries_dataset)

        # ASSERT
        self.assertEqual("tabular", res1)
        self.assertEqual("time_series", res2)


if __name__ == "__main__":
    unittest.main()
