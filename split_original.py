# 2502 + 1402 = 3904 easy ham
# 502  + 1398 = 1900 spam (* 2 = 3800)
#             =  252 hard ham

# due to data class imbalance spam examples are duplicated
# split data 80-20 train-test:
# train:
# 3120 easy ham
# 200  hard ham (may be merged with easy ham)
# 3040 spam


from os.path import isfile, isdir, join
from os import listdir, makedirs
from shutil import copyfile


def get_files(dirs):
    res = []
    for directory in dirs:
        res.extend([join(directory, f) for f in listdir(directory) if isfile(join(directory, f))])
    return res


def split_files(train_ratio=0.8):

    # getting files
    path = "original data"
    ham_dirs = ["easy_ham", "easy_ham_2", "hard_ham"]
    spam_dirs = ["spam", "spam_2"]

    ham_files = get_files(ham_dirs)
    spam_files = get_files(spam_dirs)

    if not isdir("train"):
        makedirs("train")
    if not isdir("test"):
        makedirs("test")

    # splitting
    def copy_ham(files, to):
        for file in files:
            # getting part of filename like this may cause error on windows
            # because of the '/' symbol
            copyfile(file, join(to, "ham.{}".format(file.split('/')[-1].split('.')[-1])))

    def copy_spam(files, to, duplicate=True):
        for file in files:
            copyfile(file, join(to, "spam.{}".format(file.split('/')[-1].split('.')[-1])))
            if duplicate:
                copyfile(file, join(to, "spam.{}COPY".format(file.split('/')[-1].split('.')[-1])))

    # train
    copy_ham(ham_files[:int(len(ham_files) * train_ratio)], "train")
    copy_spam(spam_files[:int(len(spam_files) * train_ratio)], "train")
    # test
    copy_ham(ham_files[int(len(ham_files) * train_ratio):], "test")
    copy_spam(spam_files[int(len(spam_files) * train_ratio):], "test", duplicate=False)
