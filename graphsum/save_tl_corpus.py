from reader import DatedTimelineCorpusReader
import argparse
import pickle

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("document_dir")
    arg_parser.add_argument("timeml_dir")
    arg_parser.add_argument("outfile")

    arg_parser.add_argument("-d", dest="doc_prefix")
    arg_parser.add_argument("-t", dest="timeml_prefix")

    args = arg_parser.parse_args()

    corpus = DatedTimelineCorpusReader(args.doc_prefix, args.timeml_prefix).run(args.document_dir, args.timeml_dir)
    with open(args.outfile, "wb") as f_out:
        pickle.dump(corpus, f_out)
