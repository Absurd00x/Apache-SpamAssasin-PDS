import re

from os.path import isfile, join
from os import listdir

# from sklearn.base import BaseEstimator, TransformerMixin
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from email import message_from_file
from email.policy import default as default_policy
from email import message
from split_original import get_files


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
# class FeatureExtractor(BaseEstimator, TransformerMixin):
class FeatureExtractor:
    # Regular expressions used while fitting and transforming
    _html_tag_re = re.compile("<[^>]*>", re.IGNORECASE)
    _valuable_word_re = re.compile("(?![^a-z])[a-z-_]{3,}", re.IGNORECASE)
    _url_re = re.compile("http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-f][0-9a-f]))+", re.IGNORECASE)
    _word_re = re.compile("(?![-=_+.@]+)[a-z-_]{2,}", re.IGNORECASE)
    _cap_word_re = re.compile("(?![-=_+.@]+)[A-Z-_]{2,}")
    _domain_re = re.compile("<?[^<]+\.[a-zA-Z]{2,3}>?", re.IGNORECASE)
    _hash_like_re = re.compile("[^\s]{20,}")
    _replied_re = re.compile(">[^\n]*")

    def __init__(self, max_dictionary_size=10, count_total_tags=True, count_total_words=True, count_total_links=True,
                 count_total_caps_words=True, count_unknown_words=True, extract_sender_domain=True,
                 extract_receiver_domain=True, dictionary_strategy="auto", words_to_search=None):
        """
        :param max_dictionary_size: Maximum number of words that can be remembered -1 if no limit
        :param count_total_tags: Whether or not should it evaluate html tags as words
        :param count_total_words: Whether or not should it count words in each mail
        :param count_total_links:  Whether or not should it count the number of links in each mail
        :param count_total_caps_words:  Whether or not should it count all caps words in each mail
        :param count_unknown_words:  Whether or not should it count all unknown words in each mail
        :param extract_sender_domain:  Whether or not should it extract sender domain
        :param extract_receiver_domain: Whether or not should it extract receiver domain
        :param dictionary_strategy: Strategy of learning words.
        "auto"   - most frequent (starting from top 20) will be searched
        "manual" - selected words will be searched. In this case "words_to_search" should be specified
        :param words_to_search: List of words that should be learnt
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

        if dictionary_strategy == "manual":
            if words_to_search is None:
                raise AttributeError("dictionary_strategy is manual, but words_to_search is not specified")
            self.max_dict_size = len(words_to_search)

        self.strategy = dictionary_strategy
        self.words_to_search = words_to_search

        # Contains "word: count" pairs
        self.vocabulary = dict()

    def _filter_mail_text(self, mail_text):
        return re.sub(self._html_tag_re, "",
                      re.sub(self._url_re, "",
                             re.sub(self._replied_re, "",
                                    re.sub(self._hash_like_re, "", mail_text))))

    def _decode_part(self, part, erase_html_tags=True):
        content_type = part.get_content_type()
        charset = part.get_content_charset()
        if charset is None:
            charset = "utf-8"
        else:
            charset = re.sub(re.compile("_?charset_?", re.IGNORECASE), "", charset)
        if charset == "default" or charset == "unknown-8bit":
            charset = "utf-8"
        if charset == "chinesebig5":
            charset = "big5"
        if content_type.lower().startswith("text"):
            part_text = part.get_payload(decode=True).decode(charset, errors="ignore")
            if content_type == "text/html":
                if erase_html_tags:
                    return re.sub(self._html_tag_re, "", part_text)
                return part_text
            if content_type.startswith("text") and not content_type.endswith("headers"):
                return part_text

    def _extract_message(self, mail, erase_html_tags=True):
        mail_text = str(mail["Subject"])
        for part in mail.walk():
            mail_text = "{}\n{}".format(mail_text, self._decode_part(part), erase_html_tags)
        return mail_text

    def _get_mail_text(self, mail_path):
        # There are non-ASCII characters which are causing errors
        mail = message_from_file(open(mail_path, encoding="utf-8", errors="ignore"), policy=default_policy)

        mail_text = self._extract_message(mail)

        return self._filter_mail_text(mail_text)

    # Construct a vocabulary
    def fit(self, X, y=None, verbose=False):
        """
        :param X: Expects list of mail file paths
        :param y: Ignore this parameter
        :param verbose: Whether or not it should print progress
        :return: self
        """
        if self.strategy == "manual":
            print("Manual strategy is chosen. Not looking through mails")
            self.vocabulary = set(self.words_to_search)
            return self
        if verbose:
            print("Started fitting mails...")

        # Learning most frequent words in lower case
        unfiltered_vocabulary = dict()

        if verbose:
            if self.max_dict_size > 0:
                print("Selecting up to {} most frequent words...".format(self.max_dict_size))
            elif self.max_dict_size == -1:
                print("Selecting all met words")

        for mail_path in X:
            text = self._get_mail_text(mail_path)
            words = set(re.findall(self._valuable_word_re, text))
            for word in words:
                word = word.lower()
                if word in unfiltered_vocabulary:
                    unfiltered_vocabulary[word] += 1
                else:
                    unfiltered_vocabulary[word] = 1

        self.vocabulary = {word for word in sorted(unfiltered_vocabulary,
                                                   key=unfiltered_vocabulary.__getitem__,
                                                   reverse=True)[20:self.max_dict_size]}
        if verbose:
            print("Words learned\nFitting finished\n")
        return self

    def _extract_domain(self, mail, prefix):
        if prefix in mail.keys():
            regexp_result = re.findall(self._domain_re, mail[prefix])
            if len(regexp_result) > 0:
                buff = regexp_result[0].strip("<>")
                buff = buff[len(buff) - 3: len(buff)]
                return buff.lstrip('.').lower()
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

        # Extracting features
        df_columns = [feature_name[feature_name.find(' ') + 1:]
                      for feature_name, collect in self.extracting_features.items() if collect]
        df_columns.extend(list(self.vocabulary))
        df_columns.append("spam")

        result = DataFrame(columns=df_columns)

        current_mail_number = 0
        current_ten_percentile = 10

        if verbose:
            print("Extracting data...")

        for mail_path in X:
            if verbose:
                if current_mail_number > len(X) * current_ten_percentile // 100:
                    print("Finished {}%".format(current_ten_percentile))
                    current_ten_percentile += 10
            feature_line = []
            mail = message_from_file(open(mail_path, encoding="utf-8", errors="ignore"), policy=default_policy)

            mail_text = self._extract_message(mail, erase_html_tags=False)

            filtered_mail = self._filter_mail_text(mail_text)

            # Adding features
            if self.extracting_features["count total tags"]:
                feature_line.append(len(re.findall(self._html_tag_re, mail_text)))

            total_words_count = len(re.findall(self._word_re, filtered_mail))
            if self.extracting_features["count total words"]:
                feature_line.append(total_words_count)

            if self.extracting_features["count total caps words"]:
                feature_line.append(len(re.findall(self._cap_word_re, filtered_mail)))

            if self.extracting_features["count total links"]:
                feature_line.append(len(re.findall(self._url_re, mail_text)))

            known_words_counts = [len(re.findall(re.compile(word, re.IGNORECASE), filtered_mail))
                                  for word in self.vocabulary]
            if self.extracting_features["count unknown words"]:
                feature_line.append(total_words_count - len(known_words_counts))

            if self.extracting_features["extract sender domain"]:
                feature_line.append(self._extract_domain(mail, "From"))

            if self.extracting_features["extract receiver domain"]:
                feature_line.append(self._extract_domain(mail, "To"))

            # Adding known words
            feature_line.extend(known_words_counts)

            # Adding labels
            feature_line.append(mail_path.split('/')[-1].split('.')[0] == "spam")

            # Adding to dataframe
            result.loc[result.shape[0]] = feature_line

            current_mail_number += 1

        if verbose:
            print("Finished transforming")

        return result

    def fit_transform(self, X, y=None, verbose=False, **fit_params):
        return self.fit(X, verbose=verbose).transform(X, verbose=verbose)


if __name__ == "__main__":
    fe = FeatureExtractor(max_dictionary_size=20)
    result = fe.fit_transform(get_files(["train"]), verbose=True)
    print(result)
    print("Here")
