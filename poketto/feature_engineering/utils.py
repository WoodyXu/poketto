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
import copy

class StringIndexer(BaseEstimator, TransformerMixin):
    """Convert categorical string to index.
    This is often the first step for categorical encoding, strongly recommend not to directly use
    the result of this class as the input to machine learning models, so we create this class just as
    a utility.
    We set the index of each categorical value based on its reversed order, 1 means the most
    frequent value, 0 means nan/null
    """
    def __init__(self, cols, topk=-1):
        """
        Parameters:
            cols: list, columns to encode, you must assign it explicitly
            topk: just encode top k values, others will be encoded as k + 1,
            encode all unique values, set -1 instead
        """
        if len(cols) == 0:
            raise ValueError("You must provide column names to be encoded!")

        self.cols = cols
        self.topk = topk
        self.mapping = []

    def fit(self, X, **kwargs):
        """
        Fit according X, we now only support dataframe.
        Paramters:
            X: a dataframe instance
        Returns:
            the instance of this class
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Now we only support pandas.DataFrame for features, please check!")

        for colname in self.cols:
            col_mapping = {"name": colname}
            data = X[colname].value_counts(ascending=False).reset_index()
            data.columns = ["value", "count"]
            for index, item in data.iterrows():
                if self.topk >= 0 and (index + 1) > self.topk:
                    break

                col_mapping.setdefault("mapping", [])
                col_mapping["mapping"].append((item.iloc[0], index + 1))

            self.mapping.append(col_mapping)

        return self

    def transform(self, X):
        """
        tranformation of string to index, we now only support dataframe
        Parameters:
            X: a dataframe instance
        Returns:
            transformed dataframe
        """
        mapping = None

        new_X = copy.deepcopy(X)

        def transform_value(x):
            if pd.isnull(x):
                return 0
            else:
                return mapping.get(x, len(mapping) + 1)

        for transform in self.mapping:
            colname = transform["name"]
            mapping = dict(transform["mapping"])
            new_X[colname + "_tmp"] = new_X[colname].map(transform_value)
            del new_X[colname]
            new_X.rename(columns={colname + "_tmp": colname}, inplace=True)

        return new_X

    @property
    def transform_mapping(self):
        return self.mapping
