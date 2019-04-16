from tlsum import GloballyClusteredSentenceCompressionTimelineGenerator
from tleval import load_corpus
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("corpus_pickles", nargs="+")
    parser.add_argument("-p", dest="use_pos", action="store_true", default=False)

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    config_basename = os.path.basename(args.config)

    for corpus_pickle in args.corpus_pickles:
        corpus = load_corpus(corpus_pickle)
        print(corpus.name)
        export(corpus, config, config_basename, args.use_pos)


def export(corpus, config, config_basename, use_pos):
    tl_gen = GloballyClusteredSentenceCompressionTimelineGenerator(config)

    if use_pos:
        out_dir = os.path.join("plain-candidate-lists-pos", config_basename)
    else:
        out_dir = os.path.join("plain-candidate-lists", config_basename)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, corpus.name.replace("/", "_") + ".txt")

    _, _, cluster_candidates = tl_gen.get_promises(corpus)

    with open(out_path, "w") as f_out:
        for candidates in cluster_candidates:
            for candidate, _ in candidates:
                if use_pos:
                    f_out.write(" ".join(map(lambda x: x[0] + "/" + x[1], candidate)))
                else:
                    f_out.write(" ".join(map(lambda x: x[0], candidate)))
                f_out.write("\n")
            f_out.write("\n")


if __name__ == '__main__':
    main()
