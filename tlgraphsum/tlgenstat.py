from tlsum import GloballyClusteredSentenceCompressionTimelineGenerator
from tleval import load_corpus

import json
import os

import argparse

from collections import defaultdict, Counter

import matplotlib.pyplot as plt

def gen_stats(args):
    all_samples = []

    for corpus_fname in args.corpora:
        for config_name in args.configs.split(","):
            with open(config_name) as f:
                config = json.load(f)
            corpus = load_corpus(corpus_fname)
            tl_gen = GloballyClusteredSentenceCompressionTimelineGenerator(config)
            samples = tl_gen.generate_date_docs_cluster_stats(corpus)

            all_samples.extend(samples)

    plt.scatter([s[0] for s in all_samples], [s[1] for s in all_samples])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs")
    parser.add_argument("corpora", nargs="+")

    args = parser.parse_args()

    cluster_lengths = Counter()
    all_cluster_factors = list()
    corpus_sizes = defaultdict(list)
    corpus_doc_counts = defaultdict(list)

    for corpus_fname in args.corpora:
        for config_name in args.configs.split(","):
            with open(config_name) as f:
                config = json.load(f)
            corpus = load_corpus(corpus_fname, False)
            tl_gen = GloballyClusteredSentenceCompressionTimelineGenerator(config)

            _, _, _, cluster_candidates = tl_gen.get_promises(corpus)
            #cluster_len = sum(len(c) for c in clusters)
            #cluster_lengths.update(len(c) for c in clusters)
            corpus_len = len(corpus.sentences)
            corpus_sizes[os.path.basename(corpus_fname).split("-")[0]].append(corpus_len)
            corpus_doc_counts[os.path.basename(corpus_fname).split("-")[0]].append(len(corpus))

            factor = sum(len(c) for c in cluster_candidates) / corpus_len
            print(factor)

            all_cluster_factors.append(factor)

    for c_name, lens in corpus_sizes.items():
        print(c_name, sum(lens) / len(lens))
    for c_name, lens in corpus_doc_counts.items():
        print(c_name, sum(lens) / len(lens))
    print(sum(all_cluster_factors) / len(all_cluster_factors))

    import sys
    sys.exit(0)


    num_cl_at_least_2 = 0
    num_cl_at_least_5 = 0
    num_cl_at_least_10 = 0

    for cl_size, cnt in sorted(cluster_lengths.items()):
        if cl_size >= 2:
            num_cl_at_least_2 += cnt

            if cl_size >= 5:
                num_cl_at_least_5 += cnt

                if cl_size >= 10:
                    num_cl_at_least_10 += cnt

    print(num_cl_at_least_2, num_cl_at_least_5, num_cl_at_least_10)


#    score_sums = defaultdict(Counter)
#
#    for corpus_fname in args.corpora:
#        for config_name in args.configs.split(","):
#            with open(config_name) as f:
#                config = json.load(f)
#            corpus = load_corpus(corpus_fname)
#            tl_gen = GloballyClusteredSentenceCompressionTimelineGenerator(config)
#            stats = tl_gen.generate_corpus_statistics(corpus)
#
#            #print(corpus_fname, "&", stats["doc_count"], "&", stats["sentence_count"], "&", stats["cluster_count"], "&", stats["candidate_count"])
#
#            score_sums[config_name]["doc_count"] += stats["doc_count"]
#            score_sums[config_name]["sentence_count"] += stats["sentence_count"]
#            score_sums[config_name]["candidate_count"] += stats["candidate_count"]
#            score_sums[config_name]["cluster_count"] += stats["cluster_count"]
#
#    for config, sums in score_sums.items():
#        print(
#            config, "&",
#            score_sums[config]["doc_count"] / len(args.corpora), "&",
#            score_sums[config]["sentence_count"] / len(args.corpora), "&",
#            score_sums[config]["cluster_count"] / len(args.corpora), "&",
#            score_sums[config]["candidate_count"] / len(args.corpora))
#