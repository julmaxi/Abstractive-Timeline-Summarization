import os
import string
from tilse.data.timelines import Timeline


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


def load_all_gold_timelines():
    all_timelines = {}
    gold_tl_dir = "gold-timelines"
    for topic_gold_tl_dir in iter_dirs(gold_tl_dir):
        for gold_tl_fname in iter_files(topic_gold_tl_dir, ".txt"):
            with open(gold_tl_fname, errors="ignore") as f:
                tl = Timeline.from_file(f)
                all_timelines[os.path.basename(topic_gold_tl_dir), os.path.basename(gold_tl_fname)] = tl

    return all_timelines
