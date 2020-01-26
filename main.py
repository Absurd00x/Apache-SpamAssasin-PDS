# 2502 + 1402 = 3904 easy ham
# 502  + 1398 = 1900 spam (* 2 = 3800)
#             =  252 hard ham

# spam is a positive class

# due to data class imbalance spam examples will be duplicated
# split data 80-20 train-test:
# train:
# 3120 easy ham
# 200  hard ham (combines with easy ham for now)
# 3040 spam

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










