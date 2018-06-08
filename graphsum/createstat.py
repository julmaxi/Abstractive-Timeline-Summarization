from reader import DateTag
import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("corpora", nargs="+")
    args = parser.parse_args()

    num_samples = 0
    num_docs = 0
    num_excact = 0

    for corpus_name in args.corpora:
        with open(corpus_name, "rb") as f:
            corpus = pickle.load(f)
        for sent in corpus.sentences:
            num_samples += len(list(filter(lambda x: x.dtype != DateTag.DAY, sent.all_date_tags)))
            num_excact += len(list(filter(lambda x: x.dtype == DateTag.DAY, sent.all_date_tags)))

        num_docs += corpus.num_documents

    print(num_docs, num_samples, num_excact)
