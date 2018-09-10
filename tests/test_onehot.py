# -*- encoding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
import pandas as pd
import unittest
from sklearn.datasets import load_boston

import poketto.feature_engineering as fe

class TestOnehotEncoder(unittest.TestCase):
    def test_normal(self):
        features = pd.DataFrame({"cat" : ["a", "b", "a", "a", "c", "b"], "num": [1, 2, 3, 4, 5, 6]})
        encoder = fe.OnehotEncoder(cols=["cat"])
        encoder.fit(features)
        features = encoder.transform(features)

        pd.testing.assert_series_equal(features["cat_1"], pd.Series([1, 0, 1, 1, 0, 0]), check_names=False)
        pd.testing.assert_series_equal(features["cat_2"], pd.Series([0, 1, 0, 0, 0, 1]), check_names=False)
        pd.testing.assert_series_equal(features["cat_3"], pd.Series([0, 0, 0, 0, 1, 0]), check_names=False)

    def test_topk(self):
        features = pd.DataFrame({"cat" : ["a", "b", "a", "a", "d", "c", "b"], 
                                 "num": [1, 2, 3, 4, 5, 6, 7]})
        encoder = fe.OnehotEncoder(cols=["cat"], topk=2)
        encoder.fit(features)
        features = encoder.transform(features)

        pd.testing.assert_series_equal(features["cat_1"], pd.Series([1, 0, 1, 1, 0, 0, 0]), check_names=False)
        pd.testing.assert_series_equal(features["cat_2"], pd.Series([0, 1, 0, 0, 0, 0, 1]), check_names=False)
        pd.testing.assert_series_equal(features["cat_3"], pd.Series([0, 0, 0, 0, 1, 1, 0]), check_names=False)

    def test_na(self):
        features = pd.DataFrame({"cat" : ["a", "b", "a", "a", "d", np.nan, "b"], 
                                 "num": [1, 2, 3, 4, 5, 6, 7]})
        encoder = fe.OnehotEncoder(cols=["cat"], topk=2)
        encoder.fit(features)
        features = encoder.transform(features)

        pd.testing.assert_series_equal(features["cat_0"], pd.Series([0, 0, 0, 0, 0, 1, 0]), check_names=False)
        pd.testing.assert_series_equal(features["cat_1"], pd.Series([1, 0, 1, 1, 0, 0, 0]), check_names=False)
        pd.testing.assert_series_equal(features["cat_2"], pd.Series([0, 1, 0, 0, 0, 0, 1]), check_names=False)
        pd.testing.assert_series_equal(features["cat_3"], pd.Series([0, 0, 0, 0, 1, 0, 0]), check_names=False)

    def test_already_index(self):
        features = pd.DataFrame({"cat" : [0, 1, 0, 0, 2, np.nan, 1],
                                 "num": [1, 2, 3, 4, 5, 6, 7]})
        encoder = fe.OnehotEncoder(cols=["cat"])
        encoder.fit(features)
        features = encoder.transform(features)

        pd.testing.assert_series_equal(features["cat_0"], pd.Series([0, 0, 0, 0, 0, 1, 0]), check_names=False)
        pd.testing.assert_series_equal(features["cat_1"], pd.Series([1, 0, 1, 1, 0, 0, 0]), check_names=False)
        pd.testing.assert_series_equal(features["cat_2"], pd.Series([0, 1, 0, 0, 0, 0, 1]), check_names=False)
        pd.testing.assert_series_equal(features["cat_3"], pd.Series([0, 0, 0, 0, 1, 0, 0]), check_names=False)

    def test_unseen(self):
        features = pd.DataFrame({"cat" : [0, 1, 0, 0, 2, np.nan, 1],
                                 "num": [1, 2, 3, 4, 5, 6, 7]})

        new_features = pd.DataFrame({"cat" : [5, 7, 0, 0, 2, np.nan, 1],
                                 "num": [1, 2, 3, 4, 5, 6, 7],
                                 "num22": [1, 2, 3, 4, 5, 6, 7]})
        encoder = fe.OnehotEncoder(cols=["cat"])
        encoder.fit(features)
        features = encoder.transform(new_features)

        pd.testing.assert_series_equal(features["cat_0"], pd.Series([0, 0, 0, 0, 0, 1, 0]), check_names=False)
        pd.testing.assert_series_equal(features["cat_1"], pd.Series([0, 0, 1, 1, 0, 0, 0]), check_names=False)
        pd.testing.assert_series_equal(features["cat_2"], pd.Series([0, 0, 0, 0, 0, 0, 1]), check_names=False)
        pd.testing.assert_series_equal(features["cat_3"], pd.Series([0, 0, 0, 0, 1, 0, 0]), check_names=False)
        pd.testing.assert_series_equal(features["cat_4"], pd.Series([1, 1, 0, 0, 0, 0, 0]), check_names=False)

    def test_realdata(self):
        bunch = load_boston()
        features = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        encoder = fe.OnehotEncoder(cols=["CHAS", "RAD"])
        encoder.fit(features)
        transformed = encoder.transform(features)

        truth_chas_1 = (features["CHAS"] == 0.0).astype(int)
        truth_chas_2 = (features["CHAS"] == 1.0).astype(int)

        pd.testing.assert_series_equal(transformed["CHAS_1"], truth_chas_1, check_names=False)
        pd.testing.assert_series_equal(transformed["CHAS_2"], truth_chas_2, check_names=False)

        truth_rad_1 = (features["RAD"] == 24.0).astype(int)
        truth_rad_2 = (features["RAD"] == 5.0).astype(int)
        truth_rad_3 = (features["RAD"] == 4.0).astype(int)
        truth_rad_4 = (features["RAD"] == 3.0).astype(int)
        truth_rad_5 = (features["RAD"] == 6.0).astype(int)
        truth_rad_6 = (features["RAD"] == 8.0).astype(int)
        truth_rad_7 = (features["RAD"] == 2.0).astype(int)
        truth_rad_8 = (features["RAD"] == 1.0).astype(int)
        truth_rad_9 = (features["RAD"] == 7.0).astype(int)

        pd.testing.assert_series_equal(transformed["RAD_1"], truth_rad_1, check_names=False)
        pd.testing.assert_series_equal(transformed["RAD_2"], truth_rad_2, check_names=False)
        pd.testing.assert_series_equal(transformed["RAD_3"], truth_rad_3, check_names=False)
        pd.testing.assert_series_equal(transformed["RAD_4"], truth_rad_4, check_names=False)
        pd.testing.assert_series_equal(transformed["RAD_5"], truth_rad_5, check_names=False)
        pd.testing.assert_series_equal(transformed["RAD_6"], truth_rad_6, check_names=False)
        pd.testing.assert_series_equal(transformed["RAD_7"], truth_rad_7, check_names=False)
        pd.testing.assert_series_equal(transformed["RAD_8"], truth_rad_8, check_names=False)
        pd.testing.assert_series_equal(transformed["RAD_9"], truth_rad_9, check_names=False)


if __name__ == "__main__":
    unittest.main()

