from pyrouge import Rouge155
from pyrouge.base import ROUGE_EVAL_HOME
from argparse import ArgumentParser

from collections import defaultdict, Counter

import os
import re


def eval_main():
    def parse_args():
        parser = ArgumentParser()
        parser.add_argument("system_dir")
        parser.add_argument("gold_dir")

        return parser.parse_args()

    args = parse_args()

    gold_summaries = read_gold_summaries(args.gold_dir)

    def summarizer(doc_id):
        fname = os.path.join(args.system_dir, "{}.sum.txt".format(doc_id))

        if os.path.isfile(fname):
            return read_summary_file(fname)

        return None

    eval_summarizer(gold_summaries, summarizer, True)


def eval_summarizer(cluster_gold_summaries, summarizer, ignore_missing=False):
    rouge = Rouge155(n_words=250, average="sentence")
    rouge_score_sums = Counter()

    num_clusters = 0
    for doc_cluster_id, gold_summaries in cluster_gold_summaries.items():
        summary = summarizer(doc_cluster_id)
        if summary is None:
            if ignore_missing:
                continue
            else:
                summary = ""

        num_clusters += 1

        scores = rouge.score_summary(summary, gold_summaries)

        rouge_score_sums += Counter(scores)

    print("Evaluated {}/{} clusters".format(num_clusters, len(cluster_gold_summaries)))
    print("SU4", rouge_score_sums["rouge_su4_f_score"] / num_clusters)
    print("R1", rouge_score_sums["rouge_1_f_score"] / num_clusters)
    print("R2", rouge_score_sums["rouge_2_f_score"] / num_clusters)




def read_gold_summaries(gold_dir):
    gold_summaries = defaultdict(dict)

    for filename in os.listdir(gold_dir):
        fname_match = re.match("D(\d+)\.M\..*\.(\w+)", filename)

        if fname_match:
            doc_id = fname_match.group(1)
            peer_id = fname_match.group(2)
            doc_content = read_summary_file(os.path.join(gold_dir, filename), encoding="latin1")

            gold_summaries[doc_id][peer_id] = doc_content

    return gold_summaries


def read_summary_file(fname, encoding="utf-8"):
    lines = []
    with open(fname, encoding=encoding) as f:
        for line in f:
            lines.append(line.split())

    return lines


if __name__ == "__main__":
    eval_main()

