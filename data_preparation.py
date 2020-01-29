import re

from os.path import isfile, join
from os import listdir

from sklearn.base import BaseEstimator, TransformerMixin
from pandas import DataFrame


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
    # Regular expressions used while fitting and transforming
    _html_tag_re = re.compile("<[^>]*>", re.IGNORECASE)
    _prefix_re = re.compile("\n(?!http|dns|ftp|url)([a-z0-9-_]+):", re.IGNORECASE)
    _valuable_word_re = re.compile(
        "(?!(?:[-=_+.@]+|are|the|and|www|net|org|com|for|any|int|edu|gov|mil|html)[^a-z])[a-z-_]{3,}", re.IGNORECASE)
    _url_re = re.compile("http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-f][0-9a-f]))+", re.IGNORECASE)
    _word_re = re.compile("(?![-=_+.@]+)[a-z-_]{2,}", re.IGNORECASE)
    _cap_word_re = re.compile("(?![-=_+.@]+)[A-Z-_]{2,}")
    _domain_re = re.compile("<?[^<]+\.[a-zA-Z]{2,3}>?", re.IGNORECASE)

    def __init__(self, max_dictionary_size=1000, count_total_tags=True, count_total_words=True, count_total_links=True,
                 count_total_caps_words=True, count_unknown_words=True, extract_sender_domain=True,
                 extract_receiver_domain=True,
                 ):
        """
        :param max_dictionary_size: Maximum number of words that can be remembered -1 if no limit
        :param count_total_tags: Whether or not should it evaluate html tags as words
        :param count_total_words: Whether or not should it count words in each mail
        :param count_total_links:  Whether or not should it count the number of links in each mail
        :param count_total_caps_words:  Whether or not should it count all caps words in each mail
        :param count_unknown_words:  Whether or not should it count all unknown words in each mail
        :param extract_sender_domain:  Whether or not should it extract sender domain
        :param extract_receiver_domain: Whether or not should it extract receiver domain
        """
        self.max_dict_size = max_dictionary_size
        self.extracting_features = {
            "count total tags": count_total_tags,
            "count total words": count_total_words,
            "count total caps words": count_total_caps_words,
            "count total links": count_total_links,
            "count unknown words": count_unknown_words,
            "extract sender domain": extract_sender_domain,
            "extract receiver domain": extract_receiver_domain
        }

        # Contains "word: count" pairs
        self.vocabulary = dict()
        # Mail prefixes after which actual message may begin
        # i.e. parameters like route, receiver, sender etc.
        self._mail_prefixes = set()

    def _get_mail_text(self, mail):
        mail_text = mail[max([mail.find(prefix) for prefix in self._mail_prefixes]) + 1:]
        return mail_text[mail_text.find('\n') + 1:].strip("\n ")

    def _extract_by_prefix(self, mail, prefix):
        prefix_index = mail.find(prefix)
        if prefix_index == -1:
            return None
        else:
            return mail[prefix_index + len(prefix) + 1: prefix_index + mail[prefix_index + 1:].find('\n') + 1]

    # Construct a vocabulary
    def fit(self, X, y=None, verbose=False):
        """
        :param X: Expects list of mail file paths
        :param y: Ignore this parameter
        :param verbose: Whether or not it should print progress
        :return: self
        """
        if verbose:
            print("Started fitting mails...")

        # Reading all the mails and keeping them in memory
        # There are non-ASCII characters which are causing errors
        mails = [open(file_path, 'r', encoding="utf-8", errors="ignore").read() for file_path in X]

        if verbose:
            print("Finding prefixes...")
        # First run
        # Finding prefixes
        mails_matches = [set(re.findall(self._prefix_re, mail)) for mail in mails]
        unfiltered_prefixes = dict()
        for mail_matches in mails_matches:
            for match in mail_matches:
                if match in unfiltered_prefixes:
                    unfiltered_prefixes[match] += 1
                else:
                    unfiltered_prefixes[match] = 1

        for prefix, count in unfiltered_prefixes.items():
            if count > 90 or prefix.startswith("X-") or prefix.startswith("List-"):
                self._mail_prefixes.add("\n{}:".format(prefix))
        del unfiltered_prefixes
        if verbose:
            print("Prefixes found")

        # Second run
        # Learning most frequent words in lower case
        unfiltered_vocabulary = dict()

        if verbose:
            print("Learning up to {} most frequent words...".format(self.max_dict_size))

        for mail in mails:
            mail_text = self._get_mail_text(mail)

            subject_index = mail.find("\nSubject:")
            if subject_index != -1:
                mail_text += '\n' + mail[subject_index + 9: mail[subject_index + 9:].find('\n')].strip()

            # Cleaning mail out of html tags
            mail_text = re.sub(self._html_tag_re, "", mail_text).strip("\n ")
            # and out of links
            mail_text = re.sub(self._url_re, "", mail_text)

            words = set(re.findall(self._valuable_word_re, mail_text))
            for word in words:
                word = word.lower()
                if word in unfiltered_vocabulary:
                    unfiltered_vocabulary[word] += 1
                else:
                    unfiltered_vocabulary[word] = 1
        self.vocabulary = {word: unfiltered_vocabulary[word] for word in sorted(unfiltered_vocabulary,
                                                                                key=unfiltered_vocabulary.__getitem__,
                                                                                reverse=True)[:self.max_dict_size]}
        if verbose:
            print("Words learned\nFitting finished\n")
        return self

    def _extract_domain(self, mail, prefix):
        value = self._extract_by_prefix(mail, prefix)
        if value is None:
            return "unknown"
        else:
            regexp_result = re.findall(self._domain_re, value)
            if len(regexp_result) > 0:
                buff = regexp_result[0].strip("<>")
                buff = buff[len(buff) - 3: len(buff)]
                return buff.lstrip('.').lower()
            else:
                return "unknown"

    # Apply regular expressions
    def transform(self, X, y=None, verbose=False):
        """
        :param X: Expects list of mail file paths
        :param y: Ignore this parameter
        :param verbose: Whether or not it should print progress
        :return: Pandas DataFrame, containing several features
        """
        if verbose:
            print("Started transforming mails...")

        mails = [open(file_path, 'r', encoding="utf-8", errors="ignore").read() for file_path in X]
        # Third run
        # Extracting features
        df_columns = [feature_name[feature_name.find(' ') + 1:]
                      for feature_name, collect in self.extracting_features.items() if collect]
        df_columns.extend([word for word in self.vocabulary])
        df_columns.append("spam")

        result = DataFrame(columns=df_columns)

        current_mail_number = 0
        current_ten_percentile = 10

        if verbose:
            print("Extracting data...")

        for full_path, mail in zip(X, mails):
            if verbose:
                if current_mail_number > int(len(X) * current_ten_percentile / 100):
                    print("Finished {}%".format(current_ten_percentile))
                    current_ten_percentile += 10
            feature_line = []
            mail_text = self._get_mail_text(mail)
            message = re.sub(self._url_re, "", re.sub(self._html_tag_re, "", mail_text)).strip("\n ")

            # Adding features
            if self.extracting_features["count total tags"]:
                feature_line.append(len(re.findall(self._html_tag_re, mail_text)))

            total_words_count = len(re.findall(self._word_re, message))
            if self.extracting_features["count total words"]:
                feature_line.append(total_words_count)

            if self.extracting_features["count total caps words"]:
                feature_line.append(len(re.findall(self._cap_word_re, message)))

            if self.extracting_features["count total links"]:
                feature_line.append(len(re.findall(self._url_re, mail_text)))

            known_words_counts = [len(re.findall(re.compile(word, re.IGNORECASE), message))
                                  for word in self.vocabulary.keys()]
            if self.extracting_features["count unknown words"]:
                feature_line.append(total_words_count - len(known_words_counts))

            if self.extracting_features["extract sender domain"]:
                feature_line.append(self._extract_domain(mail, "\nFrom:"))

            if self.extracting_features["extract receiver domain"]:
                feature_line.append(self._extract_domain(mail, "\nTo:"))

            # Adding known words
            feature_line.extend(known_words_counts)

            # Adding labels
            feature_line.append(full_path.split('/')[-1].split('.')[0] == "spam")

            # Adding to dataframe
            result.loc[result.shape[0]] = feature_line

            current_mail_number += 1

        if verbose:
            print("Finished transforming")

        return result

    def fit_transform(self, X, y=None, verbose=False, **fit_params):
        return self.fit(X, verbose=verbose).transform(X, verbose=verbose)
