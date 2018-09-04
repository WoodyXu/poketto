# -*- encoding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

class MetricsBase(object):
    """
    The abstract base metrics class
    """
    def __init__(self, truths, preds):
        """
        Parameters:
            preds: a list of guess probabilities or prediction labels
            truths: a list of ground truths
        """
        if preds is None or truths is None:
            raise ValueError("Please check None value!")
        if len(preds) == 0 or len(truths) == 0 or len(preds) != len(truths):
            raise ValueError("Lengths of predictions and truths must be equal, but we get {}"
                    " predictions and {} ground truths!".format(len(preds), len(truths)))

        # the k-v of metrics, like {'accurary': 0.8, 'recall': 0.5, ...}
        self.metrics_d = {}

    def _evaluate(self):
        raise NotImplementedError("You must inherit MetricsBase!")

    def plot(self, path_dir="/tmp/metrics", title=""):
        raise NotImplementedError("You must inherit MetricsBase!")

    @property
    def metrics(self):
        return self.metrics_d

    def __str__(self):
        raise NotImplementedError("You must inherit MetricsBase!")
