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
from sklearn.base import BaseEstimator, TransformerMixin

from .utils import StringIndexer

class OnehotEncoder(BaseEstimator, TransformerMixin):
    """
    Convert categorical feature to one hot encoding.
    """
    def __init__(self, cols, topk=-1):
        """
        Parameters:
            cols: list, columns to encode, you must assign it explicitly
            topk: just encode top k values, others will be encoded as feature_k+1,
            encode all unique values, set -1 instead
        """
        if len(cols) == 0:
            raise ValueError("You must provide column names to be encoded!")

        self.cols = cols
        self.topk = topk
        self.base_encoder = None

    def fit(self, X, **kwargs):
        self.base_encoder = StringIndexer(self.cols, self.topk)
        self.base_encoder.fit(X)

        return self

    def transform(self, X):
        new_X = self.base_encoder.transform(X)

        for colname in self.cols:
            uniques = new_X[colname].unique()
            for unique in uniques:
                new_X["{}_{}".format(colname, unique)] = (new_X[colname] == unique).astype(int)
            del new_X[colname]

        return new_X
