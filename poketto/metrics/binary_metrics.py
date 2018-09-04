# -*- encoding: utf-8 -*-

import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
plt.switch_backend("agg")

import metrics_base

class BinaryMetrics(metrics_base.MetricsBase):
    """
    Metrics Class for binary classification.
    """
    def __init__(truths, preds):
        super(BinaryMetrics, self).__init__()

        self.truths = truths.reshape(-1).astype(int)
        self.preds = preds.reshape(-1).astype(float)

        if np.min(self.preds) < 0.0 or np.max(self.preds) > 1.0:
            raise ValueError("Range of predictions must be [0, 1]!")

        unique_truths = np.unique(self.truths)
        if len(unique_truths) != 2 or 0 not in unique_truths or 1 not in unique_truths:
            raise ValueError("The ground truths for binary classification must be either 0 or 1!")

            self._evaluate()

    def _evaluate(self):
        """
        Calculate log loss, mse, auc, ks, optimal cut, 
                  average precision, accuracy, precision and recall
        """
        # auc, log loss, mse
        self.fpr, self.tpr, _ = metrics.roc_curve(self.truths, self.preds, pos_label=1)
        self.metrics_d["auc"] = metrics.auc(self.fpr, self.tpr)
        self.metrics_d["log_loss"] = metrics.log_loss(self.truths, self.preds)
        self.metrics_d["mse"] = metrics.mean_squared_error(self.truths, self.preds)

        # ks
        data = pd.DataFrame({"truth": self.truths, "prediction": self.preds})
        data["bucket"] = pd.cut(data["prediction"], np.linspace(0.0, 1.0, 1000))
        grouped = data.groupby("bucket", as_index=False)

        bucket_data = pd.DataFrame()
        bucket_data["min_pred"] = groupby.min()["prediction"]
        bucket_data["max_pred"] = groupby.max()["prediction"]
        bucket_data["positive_num"] = groupby.sum()["pos"]
        bucket_data["total_num"] = grouped.size()
        bucket_data["negative_num"] = bucket_data["total_num"] - bucket_data["positive_num"]

        bucket_data = bucket_data.sort_values(by="min_pred").reset_index(drop=True)
        bucket_data["ks"] = \
                (
                    bucket_data["negative_num"].cumsum() * 1.0 / bucket_data["negative_sum"].sum() - \
                    bucket_data["positive_num"].cumsum() * 1.0 / bucket_data["positive_num"].sum()
                ) * 100.0

        self.metrics_d["ks"] = bucket_data["ks"].max()
        opt_index = bucket_data["ks"].argmax()
        self.metrics_d["opt_cut"] = \
            (bucket_data["min_pred"].iloc[opt_index] + bucket_data["max_pred"].iloc[opt_index]) / 2.0

        # average precision, precision, recall, accuracy
        self.metrics_d["average_precision"] = metrics.average_precision_score(self.truths, self.preds)

        tp = np.sum(np.logical_and(self.preds >= self.metrics_d["opt_cut"], self.truths == 1))
        fp = np.sum(np.logical_and(self.preds >= self.metrics_d["opt_cut"], self.truths == 0))
        tn = np.sum(np.logical_and(self.preds < self.metrics_d["opt_cut"], self.truths == 0))
        fn = np.sum(np.logical_and(self.preds < self.metrics_d["opt_cut"], self.truths == 1))

        self.metrics_d["accuracy"] = (tp + tn) * 1.0 / (tp + fp + tn + fn)
        self.metrics_d["precision"] = tp * 1.0 / (tp + fp + 0.000000001)
        self.metrics_d["recall"] = tp * 1.0 / (tp + fn + 0.000000001)

    def __str__(self):
        print("The metrics are as follows: ")
        fomat_str = "AUC: {auc}, KS: {ks}, Optimal Cut: {opt_cut}\n \
                MSE: {mse},Log Loss: {log_loss}, Average Precision: {average_precision}\n \
                Accurary: {accuracy}, Precision: {precision}, Recall: {recall}"
        print(format_str.format(**self.metrics_d))

    def plot(self, path_dir="/tmp/metrics", title=""):
        """
            Plot ROC Curve, KS Curve, Precision-Recall Curve
        """
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        # ROC Curve
        plt.clf()
        plt.plot(self.fpr, self.tpr)
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.grid()
        plt.title("{} ROC Curve".format(title))
        plt.xlabel("False Positive Ratio")
        plt.ylabel("True Positive Ratio")
        plt.text(0.512, 0.512, "AUC={:.2f}".format(self.metrics_d["auc"]), fontsize=12)
        plt.savefig(os.path.join(path_dir, "roc_curve.jpg"), dpi=150)

        # KS Curve
        plt.clf()
        pos_data = self.preds[self.truths == 1]
        neg_data = self.preds[self.truths == 0]
        plt.hist(x=pos_data, bins=1000, range=(0.0, 1.0), color="blue", cumulative=True,
                histtype="step", label="Positive", density=True)
        plt.hist(x=neg_data, bins=1000, range=(0.0, 1.0), color="green", cumulative=True,
                histtype="step", label="Negative", density=True)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.title("{} KS Curve".format(title))
        plt.xlabel("Predictions")
        plt.ylabel("Cumulative Percentage")
        plt.grid()
        plt.legend(loc="upper left")

        lower_y = np.sum(pos_data <= self.metrics_d["opt_cut"]) * 1.0 / len(pos_data)
        upper_y = np.sum(neg_data <= self.metrics_d["opt_cut"]) * 1.0 / len(neg_data)
        plt.plot([self.metrics_d["opt_cut"], self.metrics_d["opt_cut"]], [lower_y, upper_y],
                color="red", linestyle="--")
        plt.text(self.metrics_d["opt_cut"] + 0.012, (lower_y + upper_y) / 2.0 + 0.012,
                "KS={:.2f}".format(self.metrics_d["ks"]), fontsize=12)
        plt.savefig(os.path.join(path_dir, "ks_curve.jpg"), dpi=150)

        # Precision-Recall Curve
        plt.clf()
        precision, recall, _ = metrics.precision_recall_curve(self.truths, self.preds, pos_label=1)
        plt.plot(recall, precision)
        plt.grid()
        plt.title("{} Precision-Recall Curve".format(title))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(os.path.join(path_dir, "pr_curve.jpg", dpi=150))
