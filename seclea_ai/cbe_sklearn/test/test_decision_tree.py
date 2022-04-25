import os
import unittest

import pandas as pd
from sklearn.model_selection import train_test_split

from ..src.decision_tree import SecleaDecisionTreeClassifier

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(base_dir, "test")


class TestDecisionTree(unittest.TestCase):
    def setUp(self) -> None:
        names = ["Class Name", "Left weight", "Left distance", "Right weight", "Right distance"]
        self.df = pd.read_csv(f"{folder_path}/balance-scale.data", names=names)
        X = self.df.drop("Class Name", axis=1)
        y = self.df[["Class Name"]]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    def test_fit(self):
        clf_model = SecleaDecisionTreeClassifier(
            criterion="gini", random_state=42, max_depth=3, min_samples_leaf=5
        )
        clf_model.fit(self.X_train, self.y_train)
        print("------------------------------------------")
        print(clf_model.max_features)


if __name__ == "__main__":
    unittest.main()
