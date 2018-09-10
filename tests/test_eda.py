# -*- encoding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import unittest
import json

import poketto.eda as eda

class TestEda(unittest.TestCase):
    def setUp(self):
        bunch = load_boston()
        X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        y = pd.Series(np.random.randint(0, 2, size=len(X)))
        numeric_cols = ["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "TAX", "PTRATIO", 
                "B", "LSTAT"]
        category_cols = ["CHAS", "RAD"]

        self.eda = eda.Eda(X=X, y=y, numeric_cols=numeric_cols, category_cols=category_cols)

    def tearDown(self):
        del self.eda

    def test_features_distribution(self):
        result = self.eda.features_distribution(plot=True)
        json.dump(result, open("./test_fd.json", "w"), indent=4)

    def test_target_distribution(self):
        result = self.eda.target_distribution(plot=True)
        json.dump(result, open("./test_td.json", "w"), indent=4)


if __name__ == "__main__":
    unittest.main()
