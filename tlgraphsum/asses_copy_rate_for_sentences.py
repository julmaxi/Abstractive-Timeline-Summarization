import csv
from tilse.data.timelines import Timeline
import os
import sys
from collections import defaultdict

from tlgraphsum.tlanalyse import CachedCorpusReader, count_tl_copied_sentences

import pathlib
import string
import datetime


def main():
    with open(sys.argv[1]) as f, open("readability-samples-fixed-withcopy.csv", "w") as f_out:
        reader = csv.reader(f)
        writer = csv.writer(f_out)

        sentences_per_system = defaultdict(list)

        corpus_basepath = pathlib.Path("./corpora")

        header = next(reader)
        writer.writerow(header + ["Copied?"])
        lines = sorted(enumerate(reader), key=lambda x: x[1][1], reverse=True)
        prev_corpus = None

        new_lines = []

        for l_idx, line in lines:
            did_copy = False
            if line[0] != "gold":
                corpus_path = corpus_basepath / (line[1] + ".pkl")
                print(corpus_path)

                corpus = CachedCorpusReader.load_corpus(str(corpus_path))
                if prev_corpus != corpus:
                    corpus_sentences = set(map(lambda s: tuple(s.as_token_attr_sequence("form_lowercase")), corpus.sentences))
                    #corpus_sentences = set(map(lambda s: "".join(s.as_token_attr_sequence("form_lowercase")).translate(str.maketrans('', '', string.punctuation)).lower(), corpus.sentences))

                #sents = line[-1].split(". ")
                #did_copy = False
                #for sent in sents:
                #    if len("".join(sent.split()).translate(str.maketrans('', '', string.punctuation)).lower()) == 0:
                #        continue
                #    if "".join(sent.split()).translate(str.maketrans('', '', string.punctuation)).lower() in corpus_sentences:
                #        did_copy = True

                tl_name = line[2]
                if "nn" not in line[0] and line[1] == "tl17-mj" and line[2] == "bbc.txt":
                    tl_name = "bbc.co.uk.txt"

                tl_path = os.path.join("filtered", "system_timelines", line[0], line[1], tl_name)
                print(tl_path)
                with open(tl_path, errors="ignore") as f:
                    tl = Timeline.from_file(f)
                    summary = tl.dates_to_summaries[datetime.date(*map(int, line[3].split("-")))]

                    for summary_line in summary:
                        summary_line = summary_line.strip()
                        if tuple(summary_line.lower().split()) in corpus_sentences:
                            print("COPY!")
                            did_copy = True

            new_line = list(line)
            new_line.append("y" if did_copy else "n")

            new_lines.append((l_idx, new_line))

        for idx, new_line in sorted(new_lines):
            writer.writerow(new_line)


if __name__ == "__main__":
    main()

