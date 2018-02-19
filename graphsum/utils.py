import os
import string


PUNCTUATION = set(string.punctuation)


def iter_files(dirname, suffix):
    for fname in os.listdir(dirname):
        if fname.endswith(suffix):
            yield os.path.join(dirname, fname)


def iter_dirs(dirname):
    for fname in os.listdir(dirname):
        fullname = os.path.join(dirname, fname)
        if os.path.isdir(fullname):
            yield fullname


def scnd(t):
    return t[1]


def fst(t):
    return t[0]


def is_punctuation(str_):
    return all(map(lambda c: c in PUNCTUATION, str_))
