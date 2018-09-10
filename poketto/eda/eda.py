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
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

class Eda(object):
    """
    The eda class
    Now this class just only supports pands.DataFrame, I will support more types later.
    """
    def __init__(self,numeric_cols, category_cols, X, y=None):
        """
        Parameters:
            X: a pandas.DataFrame instance, features data
            y: a pandas.Series instance, labels data, optional.
            numeric_cols: list, not support automatic inference numeric columns now, you must provide this
            argument, a subset of columns of X
            category_cols: list, not support automatic inference category columns now, you must provide
            this argument, a subset of columns of X 
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Now we only support pandas.DataFrame for features, please check!")

        if y is not None and not isinstance(y, pd.Series):
            raise TypeError("Now we only support pandas.Series for labels, please check!")

        self.X = X
        self.y = y

        if len(numeric_cols + category_cols) == 0:
            raise ValueError("You must at least provide column names of either numeric features or" \
                    " category features!")

        for colname in numeric_cols + category_cols:
            if colname not in self.X.columns:
                raise ValueError("{} you provide is not a column of features, please check!".format(colname))

        self.numeric_cols = numeric_cols
        self.category_cols = category_cols

    def analyze_all(self):
        """
        Automatically analyze all aspects, now include the followings:
            * features correlation
            * target and features correlation
            * features distribution
            * target distribution
        and we will plot and output the result for you.

        Returns:
        """
        self.features_correlation(plot=True, path_dir="/tmp/eda/total/features_correlation/")
        print("Features Correlation Complete. Plots in {}".format("/tmp/eda/total/features_correlation/"))

        self.target_correlation(plot=True, path_dir="/tmp/eda/total/target_correlation/")
        print("Target Correlation Complete. Plots in {}".format("/tmp/eda/total/target_correlation/"))

        self.features_distribution(plot=True, path_dir="/tmp/eda/total/features_distribution/")
        print("Features Distribution Complete. Plots in {}".format("/tmp/eda/total/features_distribution/"))

        self.target_distributionn(plot=True, path_dir="/tmp/eda/total/target_distribution/")
        print("Target Distribution Complete. Plots in {}".format("/tmp/eda/total/target_distribution/"))

    def features_correlation(self, plot=False, path_dir="", plot_title=""):
        """
        Correlations of each feature pair.
        Parameters:
            plot: whether to plot the results. If you don't need this, set False
            path_dir: if you need plotting, set this to be your own path, default
            /tmp/eda/features_correlation
            plot_title: the title prefix for each plot
        """
        pass

    def target_correlation(self, plot=False, path_dir="", plot_title=""):
        """
        Correlation of each feature and label.
        Parameters:
            plot: whether to plot the results. If you don't need this, set False
            path_dir: if you need plotting, set this to be your own path, default
            /tmp/eda/target_correlation
            plot_title: the title prefix for each plot
        """
        if self.y is None:
            raise ValueError("Please provide target column!")

    def features_distribution(self, plot=False, path_dir="", plot_title=""):
        """
        Distribution of each feature.
        Paramters:
            plot: whether to plot the results. If you don't need this, set False
            path_dir: if you need plotting, set this to be your own path, default
            /tmp/eda/features_distribution
            plot_title: the title prefix for each plot
        Returns:
            a dict, like 
            {
            "numeric": [
            {"name": "columnA", "mean":, "std":, "min":, "max":, "median":, "25qtl":, "75qtl":},
            ...
            ],

            "category": [
            {"name": "columnA", "nunique":, "topks": []}
            ]
            }
        """
        result = dict()

        if plot is True and path_dir == "":
            path_dir = "/tmp/eda/features_distribution/"
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)

        for colname in self.numeric_cols:
            ret = self._numeric_distribution(colname, plot, path_dir, plot_title) 
            result.setdefault("numeric", [])
            result["numeric"].append(ret)
            print("Finish distribution of {}".format(colname))

        for colname in self.category_cols:
            ret = self._category_distribution(colname, plot, path_dir, plot_title)
            result.setdefault("category", [])
            result["category"].append(ret)
            print("Finish distribution of {}".format(colname))

        return result

    def _numeric_distribution(self, column, plot, path_dir, plot_title):
        data = self.X[column].values.reshape(-1)

        min, max, mean, std = np.min(data), np.max(data), np.mean(data), np.std(data)
        qtl25, median, qtl75 = np.percentile(data, q=[25, 50, 75])
        ret = {"name": column, "min": min, "max": max, "mean": mean, "std": std, "25qtl": qtl25, "median": median, "75qtl": qtl75}

        if plot is True:
            plt.clf()
            plt.hist(data)
            plt.title("{} {} Distribution".format(plot_title, column))
            plt.savefig(os.path.join(path_dir, "{}.jpg".format(column)), dpi=150)

        return ret

    def _category_distribution(self, column, plot, path_dir, plot_title):
        data = pd.Series(self.X[column].values.reshape(-1))
        uniques = data.value_counts(normalize=True, ascending=False).reset_index()
        uniques.columns = ["value", "count"]

        ret = {"name": column, "nuniques": len(uniques)}

        # manually set top k
        k = 10
        uniques = uniques[: k]
        items = []
        for index, item in uniques.iterrows():
            items.append((item.iloc[0], item.iloc[1]))

        ret["topks"] = items

        if plot is True:
            plt.clf()
            plt.bar(uniques["value"], uniques["count"])
            plt.savefig(os.path.join(path_dir, "{}.jpg".format(column)), dpi=150)

        return ret

    def target_distribution(self, plot=False, path_dir="", plot_title=""):
        """
        Distribution of target.
        Parameters:
            plot: whether to plot the results. If you don't need this, set False
            path_dir: if you need plotting, set this to be your own path, default
            /tmp/eda/target_distribution
            plot_title: the title prefix for each plot
        Returns:
            a dict like:
            {
            "nuniques":,
            "values":
            [(), ()]
            }
        """
        if self.y is None:
            raise ValueError("Please provide target column!")

        result = dict()
        items = []

        if plot is True and path_dir == "":
            path_dir = "/tmp/eda/target_distribution/"
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)

        uniques = pd.Series(self.y.values.reshape(-1)).value_counts(normalize=True,
                ascending=False).reset_index()
        uniques.columns = ["value", "count"]

        result["nuniques"] = len(uniques)
        for index, item in uniques.iterrows():
            items.append((item.iloc[0], item.iloc[1]))

        result["values"] = items

        if plot is True:
            plt.clf()
            plt.bar(uniques["value"], uniques["count"])
            plt.savefig(os.path.join(path_dir, "target.jpg"), dpi=150)

        return result
