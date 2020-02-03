# spam is a positive class


from split_original import split_files, get_files
from data_preparation import FeatureExtractor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.show()


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


# fe = FeatureExtractor(max_dictionary_size=20)
# files = get_files(["train"])
# res = fe.fit_transform(files, verbose=True)
# joblib.dump(res, "dataset.pkl")
ds = joblib.load("dataset.pkl").iloc[:, :5].astype(np.int)
print(ds["total words"].sort_values(ascending=False).values[:30])

ds.hist(bins=100)
plt.show()










