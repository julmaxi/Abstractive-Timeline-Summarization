import csv
import sys
from collections import defaultdict

from tlgraphsum.tlanalyse import CachedCorpusReader, count_tl_copied_sentences

import pathlib

def main():
    with open(sys.argv[1]) as f:
        reader = csv.reader(f)

        sentences_per_system = defaultdict(list)

        corpus_basepath = pathlib.Path("./corpora")

        for l_idx, line in enumerate(reader):
            if l_idx == 0:
                continue

            corpus_path = corpus_basepath / (line[1] + ".pkl")

            corpus = CachedCorpusReader.load_corpus(str(corpus_path))
            corpus_sentences = set(map(lambda s: tuple(s.as_token_attr_sequence("form_lowercase")), corpus.sentences))

            sent = line[-1]

            sent = sent.lower()
            print(repr(sent))
            if tuple(sent.split()) in corpus_sentences:
                print("COPY!")


if __name__ == "__main__":
    main()

