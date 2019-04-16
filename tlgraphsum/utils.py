import os
import string


PUNCTUATION = set(string.punctuation)


def ensure_dir_exists(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


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


def avg(l):
    s = 0
    c = 0

    for i in l:
        s += i
        c += 1

    if c == 0:
        return 0

    return s / c
