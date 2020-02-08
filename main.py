import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_preparation import FeatureExtractor
from itertools import chain
from multiprocessing import cpu_count
from universal import get_files, parallel_processing, split_list

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)


class MonkeyNotSpamClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((X.shape[0], 1), dtype=bool).ravel()


features: pd.DataFrame
labels: pd.Series
ds = pd.read_csv("test dataset.csv")
features, labels = ds.iloc[:, :-1], ds.iloc[:, -1]

dummy_test_score = 0.5226702268038681
print("Data loaded")

models = {
    "forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "tree": DecisionTreeClassifier(random_state=42, max_depth=6, max_features=40),
    "sgd": SGDClassifier(random_state=42),
    "nn": MLPClassifier(random_state=42, hidden_layer_sizes=[50, 12], max_iter=1000, activation="relu"),
    "dummy": MonkeyNotSpamClassifier()
}


def extract_features(extractor: FeatureExtractor, train=True):
    files = get_files(["train"]) if train else get_files(["test"])
    return extractor.fit_transform(files, verbose=True)


def get_non_zero_info(words_features):
    non_zero = np.sum(np.sum(words_features != 0))
    total = words_features.shape[0] * words_features.shape[1]
    print("Total: {}\nNon-zero: {}\nRate: {:.2f}%".format(total, non_zero, non_zero * 100 / total))


def check_domains(ds, column, model):
    domains = pd.DataFrame()
    for domain in set(ds[column].values):
        series = np.where(ds[column].values == domain, 1, 0)
        score = np.mean(cross_val_score(model, series.reshape(-1, 1), labels, cv=5))
        if abs(score - dummy_test_score) > 0.01:
            print("\"{}\" is a valuable {}\tscore: {}".format(domain, column, score))
            domains[domain] = series
    return domains


def check_words(args):
    words_features, model = args
    useful_words = []
    for word in words_features.columns:
        series = words_features[word]
        score = np.mean(cross_val_score(model, series.values.reshape(-1, 1), labels, cv=5))
        if abs(score - dummy_test_score) > 0.0001:
            useful_words.append(word)
    return tuple(useful_words)


def test_models(models_dict, x, y):
    for name, model in models_dict.items():
        results = cross_val_score(model, x, y, cv=3, scoring="accuracy")
        print("{} scored {}".format(name, results))
        joblib.dump(model, "basic models/{}.pkl".format(name))


def test_model(model, x, y):
    score = np.mean(cross_val_score(model, x, y, scoring='accuracy', cv=5))
    print("Model scored {:.2f}%".format(score * 100))


def discover_useful_words(words, model=models["tree"]):
    nargs = [(part, clone(model)) for part in split_list(words, cpu_count())]
    useful_words = list(chain.from_iterable(parallel_processing(check_words, nargs)))
    words_num = len(words)
    useful_words_num = len(useful_words)
    print("useful words number is {} out of {} total => {:.2f}%".format(
        useful_words_num, words_num, useful_words_num * 100 / words_num))


def print_gridsearch_results(gs_object: GridSearchCV):
    results = gs_object.cv_results_
    to_print = list(zip(results["params"], results["mean_test_score"],
                        results["std_test_score"], results["rank_test_score"]))
    for params, mean_score, std_score, rank in sorted(to_print, key=lambda x: x[-1]):
        print("{}, {} {:.6f} {:.6f}".format(rank, params, mean_score, std_score))


nn: MLPClassifier = joblib.load("best model.pkl")
predictions = nn.predict(features)
print("Final score: {}%".format(np.sum(predictions == labels) * 100 / len(labels)))
