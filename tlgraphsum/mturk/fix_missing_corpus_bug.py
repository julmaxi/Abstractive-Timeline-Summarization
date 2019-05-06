from tilse.data.timelines import Timeline
import re
import os
import sys
import csv


def findfiles(path, regex):
    regObj = re.compile(regex)
    res = []
    for root, dirs, fnames in os.walk(path):
        for fname in fnames:
            if regObj.match(fname):
                res.append(os.path.join(root, fname))
    return res


def main():
    with open(sys.argv[1]) as f:
        reader = csv.reader(f)

        for line in reader:
            if line[0] == "gold":
                continue
            sent = line[-1]
            print(sent)
            print(findfiles("system_timelines", sent))

if __name__ == "__main__":
    main()