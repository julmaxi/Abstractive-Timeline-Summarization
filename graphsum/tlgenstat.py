from tlsum import GloballyClusteredSentenceCompressionTimelineGenerator
from tleval import load_corpus

import json

import argparse

from collections import defaultdict, Counter

import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs")
    parser.add_argument("corpora", nargs="+")

    args = parser.parse_args()

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