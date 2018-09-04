# -*- encoding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

class EdaBase(object):
    """
    The abstract base eda class
    """
    def __init__(self):
        self.insight_d = {}

    def analyze(self):
        raise NotImplementedError("You must inherit EdaBase!")

    def plot(self, path_dir="/tmp/eda", title=""):
        raise NotImplementedError("You must inherit EdaBase!")

    def __str__(self):
        if len(self.insight_d) == 0:
            self.analyze()
