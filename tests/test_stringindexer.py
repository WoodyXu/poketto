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
from sklearn.datasets import load_boston
import copy

import poketto.feature_engineering.utils as utils

class TestStringIndexer(unittest.TestCase):
    def test_normal(self):
        features = pd.DataFrame({"cat" : ["a", "b", "a", "a", "c", "b"], "num": [1, 2, 3, 4, 5, 6]})
        encoder = utils.StringIndexer(cols=["cat"])
        encoder.fit(features)

        pd.testing.assert_series_equal(encoder.transform(features)["cat"], 
                pd.Series([1, 2, 1, 1, 3, 2]), check_names=False)

    def test_topk(self):
        features = pd.DataFrame({"cat" : ["a", "b", "a", "a", "d", "c", "b"], 
                                 "num": [1, 2, 3, 4, 5, 6, 7]})
        encoder = utils.StringIndexer(cols=["cat"], topk=2)
        encoder.fit(features)

        pd.testing.assert_series_equal(encoder.transform(features)["cat"], 
                pd.Series([1, 2, 1, 1, 3, 3, 2]), check_names=False)

    def test_na(self):
        features = pd.DataFrame({"cat" : ["a", "b", "a", "a", "d", np.nan, "b"], 
                                 "num": [1, 2, 3, 4, 5, 6, 7]})
        encoder = utils.StringIndexer(cols=["cat"])
        encoder.fit(features)

        pd.testing.assert_series_equal(encoder.transform(features)["cat"], 
                pd.Series([1, 2, 1, 1, 3, 0, 2]), check_names=False)

    def test_already_index(self):
        features = pd.DataFrame({"cat" : [0, 1, 0, 0, 2, np.nan, 1], 
                                 "num": [1, 2, 3, 4, 5, 6, 7]})
        encoder = utils.StringIndexer(cols=["cat"])
        encoder.fit(features)

        pd.testing.assert_series_equal(encoder.transform(features)["cat"], 
                pd.Series([1, 2, 1, 1, 3, 0, 2]), check_names=False)

    def test_unseen(self):
        features = pd.DataFrame({"cat" : [0, 1, 0, 0, 2, np.nan, 1], 
                                 "num": [1, 2, 3, 4, 5, 6, 7]})
        encoder = utils.StringIndexer(cols=["cat"])
        encoder.fit(features)
        new_features = pd.DataFrame({"cat" : [5, 7, 0, 0, 2, np.nan, 1], 
                                 "num": [1, 2, 3, 4, 5, 6, 7],
                                 "num22": [1, 2, 3, 4, 5, 6, 7]})

        pd.testing.assert_series_equal(encoder.transform(new_features)["cat"], 
                pd.Series([4, 4, 1, 1, 3, 0, 2]), check_names=False)

    def test_realdata(self):
        bunch = load_boston()
        features = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        encoder = utils.StringIndexer(cols=["CHAS", "RAD"])
        encoder.fit(features)

        truth_chas = copy.deepcopy(features["CHAS"])
        def chas_map(x):
            if x == 0.0:
                return 1
            elif x == 1.0:
                return 2

        truth_chas = truth_chas.map(chas_map)

        truth_rad = copy.deepcopy(features["RAD"])

        def rad_map(x):
            if x == 24.0:
                return 1
            elif x == 5.0:
                return 2
            elif x == 4.0:
                return 3
            elif x == 3.0:
                return 4
            elif x == 6.0:
                return 5
            elif x == 8.0:
                return 6
            elif x == 2.0:
                return 7
            elif x == 1.0:
                return 8
            elif x == 7.0:
                return 9

        truth_rad = truth_rad.map(rad_map)

        transformed = encoder.transform(features)
        pd.testing.assert_series_equal(transformed["CHAS"],
                truth_chas, check_names=False)
        pd.testing.assert_series_equal(transformed["RAD"],
                truth_rad, check_names=False)

if __name__ == "__main__":
    unittest.main()
