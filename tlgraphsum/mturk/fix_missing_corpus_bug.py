from tilse.data.timelines import Timeline
import re
import os
import sys
import csv


def findfiles(path, regex, tl_name):
    regObj = re.compile(regex, re.IGNORECASE)
    res = []
    for root, dirs, fnames in os.walk(path):
        for fname in fnames:
            if fname != tl_name:
                continue
            with open(os.path.join(root, fname)) as f:
                if regObj.search("".join(f.read().split())):
                    res.append(os.path.join(root, fname))
    return res


def main():
    with open(sys.argv[1]) as f, open("readabiltiy-samples-fixed.csv", "w") as f_out:
        reader = csv.reader(f)
        writer = csv.writer(f_out)

        for line in reader:
            if line[-1] == "summary":
                writer.writerow(line)
                continue
            if line[0] == "gold":
                writer.writerow(line)
                continue
            sent = line[-1]
            sents = sent.split(".")
            files = []
            for s in sents:
                if len(s) > 10:
                    tl_name = line[2]
                    files.extend (findfiles("filtered/system_timelines", re.escape("".join(s.split())), tl_name))

            possible_source_corpora = set()
            for fname in files:
                cname = fname.split("/")[-2]
                possible_source_corpora.add(cname.split("+")[0])

            new_line = list(line)
            if len(possible_source_corpora) == 1:
                new_line[1] = list(possible_source_corpora)[0]
            else:
                print(sent)
                print(possible_source_corpora)
                val = input()
                new_line[1] = val
            writer.writerow(new_line)


if __name__ == "__main__":
    main()

