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


def extract_features(train=True):
    fe = FeatureExtractor(max_dictionary_size=20)
    files = get_files(["train"]) if train else get_files(["test"])
    return fe.fit_transform(files, verbose=True)


# joblib.dump(extract_features(), "dataset.pkl")
ds = joblib.load("dataset.pkl")
print(ds.describe())
print(ds.info())
print(ds.dtypes)
# TODO: Check domain influence on labels









