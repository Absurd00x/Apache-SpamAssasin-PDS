import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from data_preparation import FeatureExtractor
from itertools import chain
from multiprocessing import cpu_count
from split_original import get_files
from universal import parallel_processing, split_list, progress_measurer

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz

pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)


class MonkeyNotSpamClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((X.shape[0], 1), dtype=bool).ravel()


useful_words = joblib.load("useful words.pkl")
features: pd.DataFrame
labels: pd.Series
features, labels = joblib.load("features labels.pkl")

dummy_score = 0.5226702268038681
print("Data loaded")

models = {
    "forest": RandomForestClassifier(n_estimators=200, random_state=42, max_depth=3, n_jobs=-1),
    "tree": DecisionTreeClassifier(random_state=42, max_depth=105, max_features=71),
    "sgd": SGDClassifier(random_state=42),
    # "nn": MLPClassifier(hidden_layer_sizes=[200, 50], max_iter=2000, random_state=42),
    "dummy": MonkeyNotSpamClassifier()
}


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
    fe = FeatureExtractor(extract_receiver_domain=False, extract_sender_domain=False,
                          dictionary_strategy="manual", words_to_search=useful_words)
    files = get_files(["train"]) if train else get_files(["test"])
    return fe.fit_transform(files, verbose=True)


def get_non_zero_info(words_features):
    non_zero = np.sum(np.sum(words_features != 0))
    total = words_features.shape[0] * words_features.shape[1]
    print("Total: {}\nNon-zero: {}\nRate: {:.2f}%".format(total, non_zero, non_zero * 100 / total))


def check_domains(ds, column, model):
    domains = pd.DataFrame()
    for domain in set(ds[column].values):
        series = np.where(ds[column].values == domain, 1, 0)
        score = np.mean(cross_val_score(model, series.reshape(-1, 1), labels, cv=5))
        if abs(score - dummy_score) > 0.01:
            print("\"{}\" is a valuable {}\tscore: {}".format(domain, column, score))
            domains[domain] = series
    return domains


def check_words(args):
    words_features, model = args
    useful_words = []
    for word in words_features.columns:
        series = words_features[word]
        score = np.mean(cross_val_score(model, series.values.reshape(-1, 1), labels, cv=5))
        if abs(score - dummy_score) > 0.0001:
            # print("\"{}\" is a valuable word\t score: {}".format(word, score))
            useful_words.append(word)
    return tuple(useful_words)


def test_models(models_dict, x, y):
    for name, model in models_dict.items():
        results = cross_val_score(model, x, y, cv=3, scoring="accuracy")
        print("{} scored {}".format(name, results))
        joblib.dump(model, "basic models/{}.pkl".format(name))


def test_model(model, x, y):
    score = np.mean(cross_val_score(model, x, y, scoring='accuracy', cv=5))
    print("Model scored {:.2f}% using data from above".format(score))


def discover_useful_words(words, model=models["tree"]):
    nargs = [(part, clone(model)) for part in split_list(words, cpu_count())]
    useful_words = list(chain.from_iterable(parallel_processing(check_words, nargs)))
    words_num = len(words)
    useful_words_num = len(useful_words)
    print("useful words number is {} out of {} total => {:.2f}%".format(
        useful_words_num, words_num, useful_words_num * 100 / words_num))
    # joblib.dump(useful_words, "useful words.pkl")


def print_gridsearch_results(gs_object: GridSearchCV):
    results = gs_object.cv_results_
    to_print = list(zip(results["params"], results["mean_test_score"],
                        results["std_test_score"], results["rank_test_score"]))
    for params, mean_score, std_score, rank in sorted(to_print, key=lambda x: x[-1]):
        print("{}, {} {:.6f} {:.6f}".format(rank, params, mean_score, std_score))


# ds = extract_features(train=False)
# joblib.dump((ds.iloc[:, :-1], ds.iloc[:, -1]), "test features labels.pkl")

model: DecisionTreeClassifier = joblib.load("best model.pkl")

feature_importance = dict(zip(features.columns, model.feature_importances_))
for feature in sorted(feature_importance.keys(), key=feature_importance.__getitem__, reverse=True):
    print("{}:\t{}".format(feature, feature_importance[feature]))
