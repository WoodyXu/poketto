# -*- encoding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
import unittest

import poketto.metrics as metrics 

class TestBinaryMetrics(unittest.TestCase):
    def test_invalid_input(self):
        self.assertRaises(ValueError, metrics.BinaryMetrics, None, [1])
        self.assertRaises(ValueError, metrics.BinaryMetrics, [1], None)
        self.assertRaises(ValueError, metrics.BinaryMetrics, [1, 0], [0.4, 0.3, 0.1])
        self.assertRaises(ValueError, metrics.BinaryMetrics, [1, 0, 2], [0.4, 0.3, 0.1])
        self.assertRaises(ValueError, metrics.BinaryMetrics, [1, 0, 1], [1.2, 0.3, 0.1])

    def test_auc(self):
        y = [0, 0, 1, 1]
        pred = [0.1, 0.4, 0.35, 0.8]
        metric = metrics.BinaryMetrics(y, pred)
        self.assertAlmostEqual(metric.metrics["auc"], 0.75)

    def test_average_precision(self):
        y = [0, 0, 1, 1]
        pred = [0.1, 0.4, 0.35, 0.8]
        metric = metrics.BinaryMetrics(y, pred)
        self.assertAlmostEqual(metric.metrics["average_precision"], 0.83333333)

    def test_logloss(self):
        y = [0, 1, 1, 0]
        pred = [0.1, 0.9, 0.8, 0.35]
        metric = metrics.BinaryMetrics(y, pred)
        self.assertAlmostEqual(metric.metrics["logloss"], 0.21616187)

    def test_mse(self):
        pass

    def test_ks(self):
        y = np.random.randint(0, 2, 10000)
        pred = np.random.rand(10000)
        metric = metrics.BinaryMetrics(y, pred)
        self.assertAlmostEqual(metric.metrics["ks"], np.max(metric.tpr - metric.fpr), places=3)
        self.assertAlmostEqual(metric.metrics["opt_cut"], 
            metric.threshold[np.argmax(metric.tpr - metric.fpr)], places=3)
        metric.plot()


if __name__ == "__main__":
    unittest.main()
