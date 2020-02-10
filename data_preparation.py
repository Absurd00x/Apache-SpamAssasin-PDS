import re

from collections import Counter
from email import message_from_file
from email.policy import default as default_policy
from itertools import chain
from multiprocessing import cpu_count
from numpy import int64
from pandas import DataFrame, concat
from sklearn.base import BaseEstimator, TransformerMixin
from universal import get_files, parallel_processing, split_list

# Regular expressions used while fitting and transforming
regexps = {
    "html tag": re.compile("<[^>]*>", re.IGNORECASE),
    "valuable word": re.compile("[a-z-_']{3,}", re.IGNORECASE),
    "url": re.compile("http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-f][0-9a-f]))+", re.IGNORECASE),
    "word": re.compile("(?![-=_+.@]+)[a-z-_]{2,}", re.IGNORECASE),
    "cap word": re.compile("(?![-=_+.@]+)[A-Z-_]{2,}"),
    "domain": re.compile("<?[^<]+\.[a-zA-Z]{2,3}>?", re.IGNORECASE),
    "hash-like": re.compile("[^\s]{20,}"),
    "replied": re.compile(">[^\n]*"),
}


def filter_mail_text(mail_text):
    return re.sub(regexps["html tag"], "",
                  re.sub(regexps["url"], "",
                         re.sub(regexps["replied"], "",
                                re.sub(regexps["hash-like"], "", mail_text))))


def decode_part(part, erase_html_tags=True):
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
                return re.sub(regexps["html tag"], "", part_text)
            return part_text
        if content_type.startswith("text") and not content_type.endswith("headers"):
            return part_text
    return ''


def extract_message(mail, erase_html_tags=True):
    subject = str(mail["Subject"])
    text = '\n'.join([decode_part(part, erase_html_tags) for part in mail.walk()])
    return "{}\n{}".format(subject, text)


def read_mail(mail_path):
    with open(mail_path, encoding="utf-8", errors="ignore") as file:
        return message_from_file(file, policy=default_policy)


def get_mail_data(mail_path, erase_html_tags=True):
    # There are non-ASCII characters which are causing errors
    mail = read_mail(mail_path)
    mail_text = extract_message(mail, erase_html_tags)
    filtered_text = filter_mail_text(mail_text)

    return mail, mail_text, filtered_text


def extract_domain(mail, prefix):
    if prefix in mail.keys():
        regexp_result = re.findall(regexps["domain"], mail[prefix])
        if len(regexp_result) > 0:
            buff = regexp_result[0].strip("<>")
            buff = buff[len(buff) - 3: len(buff)]
            return buff.lstrip('.').lower()
    return "unknown"


def learn_words_from_mails(mails_paths):
    words_in_mails = [frozenset(re.findall(regexps["valuable word"], get_mail_data(mail_path)[2].lower()))
                      for mail_path in mails_paths]
    cleared_words_in_mails = [word.strip("'_-") for word in chain.from_iterable(words_in_mails)
                              if word.strip("'_-") != '' and len(word.strip("'_-")) > 2]
    return Counter(cleared_words_in_mails)


def mails_to_features(args):
    mails_paths, vocabulary, features, df = args

    for mail_path in mails_paths:
        feature_line = []
        mail, mail_text, filtered_mail = get_mail_data(mail_path, False)

        # Adding features
        if features["count total tags"]:
            feature_line.append(len(re.findall(regexps["html tag"], mail_text)))

        total_words_count = len(re.findall(regexps["word"], filtered_mail))
        if features["count total words"]:
            feature_line.append(total_words_count)

        if features["count total caps words"]:
            feature_line.append(len(re.findall(regexps["cap word"], filtered_mail)))

        if features["count total links"]:
            feature_line.append(len(re.findall(regexps["url"], mail_text)))

        known_words_counts = [len(re.findall(re.compile(word, re.IGNORECASE), filtered_mail)) for word in vocabulary]
        if features["count unknown words"]:
            feature_line.append(total_words_count - sum(known_words_counts))

        if features["extract sender domain"]:
            feature_line.append(extract_domain(mail, "From"))

        if features["extract receiver domain"]:
            feature_line.append(extract_domain(mail, "To"))

        # Adding known words
        feature_line.extend(known_words_counts)
        # Adding labels
        feature_line.append(mail_path.split('/')[-1].split('.')[0] == "spam")
        # Adding to DataFrame
        df.loc[df.shape[0]] = feature_line

    return df


# Extracts features on the top
class FeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, max_dictionary_size=10, count_total_tags=True, count_total_words=True, count_total_links=True,
                 count_total_caps_words=True, count_unknown_words=True, extract_sender_domain=True,
                 extract_receiver_domain=True, dictionary_strategy="auto", words_to_search=None, n_jobs=cpu_count()):
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
        :param n_jobs: number of available CPUs
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
                raise AttributeError("\"dictionary_strategy\" is manual, but words_to_search is not specified")
            self.max_dict_size = len(words_to_search)

        self.strategy = dictionary_strategy
        self.words_to_search = words_to_search
        if n_jobs < 1 or not isinstance(n_jobs, int):
            raise AttributeError("\"n_jobs\" must be a positive integer")
        self.n_jobs = n_jobs

        # Contains learned words
        if self.strategy == "auto":
            self.vocabulary = None
        elif self.strategy == "manual":
            self.vocabulary = tuple(self.words_to_search)
        else:
            response = "Strategy \"{}\" is not implemented\nUse either \"auto\" or \"manual\""
            raise AttributeError(response.format(self.strategy))

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
            return self
        if verbose:
            print("Started fitting mails...")
            if self.max_dict_size > 0:
                print("Selecting up to {} most frequent words...".format(self.max_dict_size))
            elif self.max_dict_size == -1:
                print("Selecting all met words")

        # Learning most frequent words in lower case
        if self.n_jobs == 1:
            vocabulary = learn_words_from_mails(X)
        else:
            results = parallel_processing(learn_words_from_mails, split_list(X, self.n_jobs))
            vocabulary = dict()
            for dictionary in results:
                vocabulary.update(dictionary)

        if self.max_dict_size == -1:
            self.vocabulary = tuple(sorted(vocabulary.keys(), key=vocabulary.__getitem__, reverse=True))
            if verbose:
                print("Actual vocabulary size is", len(self.vocabulary))
        self.vocabulary = tuple(sorted(vocabulary.keys(), key=vocabulary.__getitem__,
                                       reverse=True)[:self.max_dict_size])
        if verbose:
            print("Words learned\nFitting finished\n")
        return self

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
        df_columns.extend(self.vocabulary)
        df_columns.append("spam")

        result = DataFrame(columns=df_columns).astype(int64)

        result["spam"] = result["spam"].astype(bool)

        if verbose:
            print("Extracting information from data...")

        if self.n_jobs == 1:
            result = mails_to_features((X, self.vocabulary, self.extracting_features, result))
        else:
            nargs = [(mails_chunk, self.vocabulary, self.extracting_features, result.copy(deep=True))
                     for mails_chunk in split_list(X, self.n_jobs)]
            # Extracting features by blocks
            dataframes = parallel_processing(mails_to_features, nargs)
            # Adding to DataFrame
            for dataframe in dataframes:
                result = concat([result, dataframe])

        # Converting to appropriate types
        if self.extracting_features["extract sender domain"]:
            result["sender domain"] = result["sender domain"].astype('category')

        if self.extracting_features["extract receiver domain"]:
            result["receiver domain"] = result["receiver domain"].astype('category')

        if verbose:
            print("Finished transforming")

        return result

    def fit_transform(self, X, y=None, verbose=False, **fit_params):
        return self.fit(X, verbose=verbose).transform(X, verbose=verbose)


if __name__ == "__main__":
    fe = FeatureExtractor(max_dictionary_size=20)
    test = fe.fit_transform(get_files(["train"]), verbose=True)
    print(test)
    print("Test complete")
