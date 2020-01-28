import re

from os.path import isfile, join
from os import listdir

from sklearn.base import BaseEstimator, TransformerMixin
from pandas import DataFrame, Series


def get_files(dirs):
    res = []
    for dir in dirs:
        res.extend([join(dir, f) for f in listdir(dir) if isfile(join(dir, f))])
    return res


# features                          | interpretation  |
# ----------------------------------|-----------------|
# from (domain)                     | binary
# word (including tags) : count     | binary: cont
# total word num                    | cont
# caps word num                     | cont
# to (domain)                       | binary
# unk word: count                   | binary: cont
# num of links                      | cont
# ----------------------------------|----------------|


# Extracts features on the top 
class FeatureExtractor(BaseEstimator, TransformerMixin):

    # Contains "word: count" pairs
    vocabulary = dict()
    # Mail prefixes after which actual message may begin
    # i.e. parameters like route, receiver, sender etc.
    _mail_prefixes = set()

    def __init__(self, max_dictionary_size=100, collect_tags=True, count_total_words=True,
                 count_total_caps_words=True, extract_sender_domain=True, extract_receiver_domain=True,
                 count_links=True):
        """
        :param max_dictionary_size: Maximum number of words that can be remembered -1 if no limit
        :param collect_tags: Whether or not should it evaluate html tags as words 
        :param count_total_words: Whether or not should it count words in each mail
        :param count_total_caps_words:  Whether or not should it count all caps words in each mail 
        :param extract_sender_domain:  Whether or not should it extract sender domain
        :param extract_receiver_domain: Whether or not should it extract receiver domain
        :param count_links:  Whether or not should it count the number of links in each mail
        """
        self.max_dict_size = max_dictionary_size
        self.collect_tags = collect_tags

    # Construct a vocabulary
    def fit(self, X, y=None):
        """
        :param X: Expects list of mail file paths
        :param y: Ignore this parameter
        :return: Nothing
        """

        # Reading all the mails and keeping them in memory
        # There are non-ASCII characters which are causing errors
        mails = [open(file_path, 'r', encoding="utf-8", errors="ignore").read() for file_path in X]

        # First run
        # Finding prefixes
        regexp = re.compile("\n(?!http|dns|ftp|url)([a-z0-9-_]+):", re.IGNORECASE)
        mails_matches = [set(re.findall(regexp, mail)) for mail in mails]
        unfiltered_prefixes = dict()
        for mail_matches in mails_matches:
            for match in mail_matches:
                if match in unfiltered_prefixes:
                    unfiltered_prefixes[match] += 1
                else:
                    unfiltered_prefixes[match] = 1

        for prefix, count in unfiltered_prefixes.items():
            if count > 90 or prefix.startswith("X-"):
                self._mail_prefixes.add("\n{}:".format(prefix))

        # Second run
        # Learning most frequent words in lower case
        re.compile("[a-z-_]+")
        for mail in mails:
            mail_text = mail[max([mail.find(prefix) for prefix in self._mail_prefixes]) + 1:]
            mail_text = mail_text[mail_text.find('\n') + 1:].strip("\n ")
            lower_case_mail_text = mail_text.lower()





    # Apply regular expressions
    def transform(self, X, y=None):
        """
        :param X: Expects list of mail file paths
        :param y: Ignore this parameter
        :return: Pandas DataFrame, containing several features
        """
        # here goes the code
        pass


fe = FeatureExtractor()
files = get_files(["train"])
fe.fit(files)











