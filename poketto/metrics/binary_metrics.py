# -*- encoding: utf-8 -*-

import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import metrics_base

class BinaryMetrics(metrics_base.MetricsBase):
    """
    Metrics Class for binary classification.
    """
    def __init__(preds, truths):
        super(BinaryMetrics, self).__init__()


