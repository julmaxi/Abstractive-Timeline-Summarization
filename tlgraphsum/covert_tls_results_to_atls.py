from collections import defaultdict

import sys
import os
import pickle

import argparse
from tilse.data.timelines import Timeline, GroundTruth
from tilse.evaluation import rouge
import shutil
from tlgraphsum.tleval import write_results_file


def read_tls_results_file(fname):
    current_metric = None
    current_sub_metric = None

    results = defaultdict(lambda: defaultdict(dict))

    with open(fname) as f:
        for line in f:
            line = line.strip()

            if len(line) == 0:
                current_metric = None
                current_sub_metric = None
                continue

            if line[0] == "-":
                continue

            if line == "date selection":
                current_metric = "date"
                continue
            elif line in ("rouge_1", "rouge_2"):
                current_metric = line
                continue

            if line in ("concat", "align_date_content_costs_many_to_one", "agreement"):
                current_sub_metric = line
                continue

            if current_metric is not None:
                print(line)
                tl, _, prec, _, rec, _, f1 = line.split()

                if tl == "average_score":
                    continue

                corpus, tlidx = tl.split("_", 1)
                results[corpus][tlidx][(current_metric, current_sub_metric)] = tuple(map(float, (prec, rec, f1)))

    return results


def write_results_file_score_only(fname, results):
    metrics_order = [
                ("date", None),
                ("rouge_1", "concat"),
                ("rouge_2", "concat"),
                ("rouge_1", "agreement"),
                ("rouge_2", "agreement"),
                ("rouge_1", "align_date_content_costs_many_to_one"),
                ("rouge_2", "align_date_content_costs_many_to_one")
    ]
    with open(fname, "w") as f:
        f.write("Timeline\tDate R\tDate P\tDate F1\tR1 R\tR1 P\tR1 F1\tR2 R\tR2 P\tR2 F1\tR1 R\tR1 P\tR1 F1\tR2 R\tR2 P\tR2 F1\tR1 R\tR1 P\tR1 F1\tR2 R\tR2 P\tR2 F1\n")
        metric_sums = defaultdict(lambda: [0, 0, 0])
        for tl_idx, tl in sorted(results.items()):
            f.write(str(tl_idx) + "\t")
            for metric in metrics_order:
                vals = results[tl_idx][metric]
                f.write("{:.4f}\t{:.4f}\t{:.4f}".format(*[val for val in vals]))

                metric_sums[metric][0] += vals[0]
                metric_sums[metric][1] += vals[1]
                metric_sums[metric][2] += vals[2]

                if metric == ("rouge_2", "align_date_content_costs_many_to_one"):
                    f.write("\n")
                else:
                    f.write("\t")
        f.write("All\t")

        for idx, metric in enumerate(metrics_order):
            vals = metric_sums[metric]
            f.write("{:.4f}\t{:.4f}\t{:.4f}".format(*[val / len(results) for val in vals]))

            if idx == len(metrics_order):
                f.write("\n")
            else:
                f.write("\t")


def read_tls_results_obj_file(fname):
    with open(fname, "rb") as f:
        sys_timelines, gold_timelines, scores = pickle.load(f)
    results = defaultdict(lambda: defaultdict(dict))

    for tl_identifier, rouge_scores in scores.mapping.items():
        if tl_identifier == "average_score":
            continue
        corpus, tlidx = tl_identifier.split("_", 1)
        for score_function_type, type_rouge_scores in rouge_scores.items():
            for rouge_score_type, rouge_score in type_rouge_scores.items():
                prec, rec, f1 = rouge_score["precision"], rouge_score["recall"], rouge_score["f_score"]

                print(rec, prec, f1)
                results[corpus][tlidx][(rouge_score_type, score_function_type)] = tuple(map(float, (prec, rec, f1)))
    for tl_identifier, date_score in scores.date_mapping.items():
        if tl_identifier == "average_score":
            continue
        corpus, tlidx = tl_identifier.split("_", 1)
        prec, rec, f1 = date_score["precision"], date_score["recall"], date_score["f_score"]
        results[corpus][tlidx][("date", None)] = (prec, rec, f1)

    return results


def read_tls_timelines_obj_file(fname):
    with open(fname, "rb") as f:
        sys_timelines, gold_timelines, scores = pickle.load(f)

    keys_sys = sorted(sys_timelines)
    keys_gold = sorted(gold_timelines)

    timelines_per_corpus = defaultdict(list)

    for key_gold, key_sys in zip(keys_gold, keys_sys):
        tl_gold = gold_timelines[key_gold]
        tl_sys = sys_timelines[key_sys]

        corpus_name, tl_name = key_gold.split("_", 1)

        timelines_per_corpus[corpus_name].append((tl_name, tl_gold, tl_sys))

    return timelines_per_corpus




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file")
    parser.add_argument("system_name")
    parser.add_argument("prefix")
    args = parser.parse_args()

    per_corpus_tls = read_tls_timelines_obj_file(args.results_file)

    for corpus_name, timelines in per_corpus_tls.items():
        modified_corpus_name = args.prefix + corpus_name

        target_eval_dir = os.path.join("evaluation_results", args.system_name)
        target_eval_path = os.path.join(target_eval_dir, modified_corpus_name + ".txt")
        target_timelines_dir = os.path.join("system_timelines", args.system_name, modified_corpus_name)

        if not os.path.isdir(target_timelines_dir):
            os.makedirs(target_timelines_dir)
        if not os.path.isdir(target_eval_dir):
            os.makedirs(target_eval_dir)

        print(target_eval_path)
        print(target_timelines_dir)

        gold_tls_and_names = [(name, tl) for name, tl, _ in timelines]
        sys_tls = [tl for _, _, tl in timelines]

        write_results_file(target_eval_path, target_timelines_dir, gold_tls_and_names, sys_tls)

    sys.exit(0)

    write_results_file(os.path.join(system_evaluation_dir, corpus_name[:-len(".pkl")] + ".txt"), corpus_sys_tl_dir, gold_tls, sys_tls)


    # all_results = read_tls_results_obj_file(args.results_file)

    # if not os.path.isdir(os.path.join(args.outdir, args.system_name)):
    #     os.mkdir(os.path.join(args.outdir, args.system_name))

    # for corpus, corpus_results in all_results.items():
    #     corpus_out_fname = os.path.join(args.outdir, args.system_name, args.prefix + corpus + ".txt")
    #     write_results_file(corpus_out_fname, corpus_results)
