import argparse
from tilse.data.timelines import Timeline
from tleval import load_corpus
from tlanalyse import analyse_results_file
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import scipy


def read_results_dir(dirname):
    all_results = defaultdict(dict)
    for fname in os.listdir(dirname):
        if not fname.endswith(".txt"):
            continue

        topic_results = analyse_results_file(os.path.join(dirname, fname))
        for tl_name, result in topic_results.items():
            all_results[os.path.basename(fname)][tl_name] = result

    return all_results


def compute_spread(gold_tl):
    tl_start = min(gold_tl.dates_to_summaries)
    tl_end = max(gold_tl.dates_to_summaries)
    tl_dates = (tl_end - tl_start).days + 1

    print(len(gold_tl.dates_to_summaries) / tl_dates)

    return len(gold_tl.dates_to_summaries) / tl_dates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sys_1_results_dir")
    parser.add_argument("sys_2_results_dir")

    args = parser.parse_args()

    results_1 = read_results_dir(args.sys_1_results_dir)
    results_2 = read_results_dir(args.sys_2_results_dir)
    score_diffs = []
    available_sents = []
    compression_rates = []
    spreads = []

    for corpus_name in results_1:
        if corpus_name not in results_2:
            continue

        corpus = load_corpus("corpora/" + corpus_name.rsplit(".")[0] + ".pkl")

        for tl_name, result_1 in results_1[corpus_name].items():
            result_2 = results_2[corpus_name][tl_name]
            with open("gold-timelines/" + corpus_name.split(".")[0] + "/" + tl_name, errors="ignore") as f:
                print("gold-timelines/" + corpus_name.split(".")[0] + "/" + tl_name)
                gold_tl = Timeline.from_file(f)

            total_tl_length = sum(map(len, gold_tl.dates_to_summaries.values()))
            total_corpus_length = len(corpus.sentences)

            score_diffs.append(result_1.rouge_2_align.f1 - result_2.rouge_2_align.f1)
            available_sents.append(len(corpus.docs_between_dates(min(gold_tl.get_dates()), max(gold_tl.get_dates()))))
            compression_rates.append(total_tl_length / total_corpus_length)

            spreads.append(compute_spread(gold_tl))

    print("Sents", scipy.stats.spearmanr(available_sents, score_diffs))
    print("Compression", scipy.stats.spearmanr(compression_rates, score_diffs))
    print("Spread", scipy.stats.spearmanr(spreads, score_diffs))

    plt.axhline(color="b")
    plt.scatter(available_sents, score_diffs, c=["r" if score_diff <= 0.0 else "b" for score_diff in score_diffs])
    plt.figure()
    plt.axhline(color="b")
    plt.scatter(compression_rates, score_diffs, c=["r" if score_diff <= 0.0 else "b" for score_diff in score_diffs])
    plt.figure()
    plt.axhline(color="b")
    plt.scatter(spreads, score_diffs, c=["r" if score_diff <= 0.0 else "b" for score_diff in score_diffs])

    plt.show()


        #load_corpus(os.path.join(args.corpus_dir))
if __name__ == "__main__":
    main()
