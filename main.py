# spam is a positive class

# features        | interpretation|
# ---------------------------------
# from (domain)   | binary
# word: count     | binary: cont
# total word num  | cont
# caps word num   | cont
# to (domain)     | binary
# unk word: count | binary: cont
# num of links    | cont
# --------------------------------|


import matplotlib.pyplot as plt
from split_original import split_files


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


split_files()










