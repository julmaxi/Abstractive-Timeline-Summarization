import os


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

