from tlsum import GloballyClusteredSentenceCompressionTimelineGenerator
from tleval import load_corpus
import argparse
import json
import os

import random

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("corpora", nargs="+")
    parser.add_argument("outfile")

    args = parser.parse_args()

    if os.path.isfile(args.outfile):
        print("WARNING: file {} exists. Continue (Y/n)".format(args.outfile))
        if input() == "n":
            return

    with open(args.config) as f:
        config = json.load(f)

    flat_candidates = []
    all_sentences = []
    for corpus_name in args.corpora:
        corpus = load_corpus(corpus_name)

        tl_gen = GloballyClusteredSentenceCompressionTimelineGenerator(config)
        tl_gen.generator.prepare(corpus)

        clusters = tl_gen.create_clusters(corpus)
        cluster_candidates = tl_gen.generate_candidates_for_clusters(corpus, clusters)

        for candidates in cluster_candidates:
            for candidate, _ in candidates:
                flat_candidates.append(candidate)

        all_sentences.extend(corpus.sentences)

    num_samples = min(len(all_sentences), len(flat_candidates))

    sentence_samples = random.sample(all_sentences, num_samples)
    candidate_samples = random.sample(flat_candidates, num_samples)
    sentence_sample_lens = []
    candidate_sample_lens = []

    with open(args.outfile, "w") as f_out:
        for sent_sam in sentence_samples:
            f_out.write(":S\t")
            f_out.write(sent_sam.as_tokenized_string_with_attribute("pos"))
            f_out.write("\n")
            sentence_sample_lens.append(len(sent_sam))
        for cand_sam in candidate_samples:
            f_out.write(":C\t")
            f_out.write(" ".join(map(lambda x: x[0] + "/" + x[1], cand_sam)))
            f_out.write("\n")
            candidate_sample_lens.append(len(cand_sam))

    print(np.array(sentence_sample_lens).avg(), np.array(sentence_sample_lens).std())
    print(np.array(candidate_sample_lens).avg(), np.array(candidate_sample_lens).std())


if __name__ == '__main__':
    main()
