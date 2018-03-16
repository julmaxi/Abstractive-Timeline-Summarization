from tilse.data.timelines import Timeline, GroundTruth
from tilse.evaluation import rouge
import argparse

from tlsum import create_timeline_sentence_level, TimelineParameters, APClusteringTimelineGenerator, GloballyClusteredSentenceCompressionTimelineGenerator

import pickle

import json
import os


def determine_tl_parameters(timeline):
    dateset = timeline.get_dates()
    earliest_date = min(*dateset)
    last_date = max(*dateset)
    tl_date_count = len(dateset)

    max_date_sent_count = 0
    total_sent_len = 0
    for date in timeline:
        sents = timeline[date]
        total_sent_len += len(sents)

        max_date_sent_count = max(max_date_sent_count, len(sents))

    return TimelineParameters(
        earliest_date,
        last_date,
        tl_date_count,
        total_sent_len,
        max_date_sent_count
    )


def load_corpus(fname):
    with open(fname, "rb") as f:
        corpus = pickle.load(f)
        corpus.name = fname

    return corpus



def evaluate_tl_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", dest="timelines", nargs="+")
    parser.add_argument("corpus_pickle")
    parser.add_argument("config")

    args = parser.parse_args()

    evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])

    rouge_1_sum = 0
    rouge_1_r_sum = 0
    rouge_1_p_sum = 0
    rouge_2_sum = 0
    rouge_2_r_sum = 0
    rouge_2_p_sum = 0

    align_rouge_1_sum = 0
    align_rouge_1_r_sum = 0
    align_rouge_1_p_sum = 0
    align_rouge_2_sum = 0
    align_rouge_2_r_sum = 0
    align_rouge_2_p_sum = 0

    date_f1_sum = 0
    date_f1_r_sum = 0
    date_f1_p_sum = 0

    corpus = load_corpus(args.corpus_pickle)

    timelines = []

    for tl_fname in args.timelines:
        with open(tl_fname, encoding="latin-1") as f:
            timeline = Timeline.from_file(f)
            timelines.append((os.path.basename(tl_fname), timeline))

    #tl_gen = APClusteringTimelineGenerator(True)

    with open(args.config) as f:
        config = json.load(f)

    tl_gen = GloballyClusteredSentenceCompressionTimelineGenerator(config)

    corpus_basename = os.path.basename(args.corpus_pickle).split(".")[0]
    config_basename = os.path.basename(args.config)

    out_timelines_dir = os.path.join("system_timelines", config_basename, corpus_basename)
    results_dir = os.path.join("evaluation_results", config_basename)

    if not os.path.isdir(out_timelines_dir):
        os.makedirs(out_timelines_dir)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    sys_timelines = tl_gen.generate_timelines(corpus, [determine_tl_parameters(tl) for _, tl in timelines], reference_timelines=list(map(lambda x: x[1], timelines)))
    with open(os.path.join(results_dir, corpus_basename + ".txt"), "w") as f_out: 
        f_out.write("Timeline        \tDate R\tDate P\tDate F1\tR1 R\tR1 P\tR1 F1\tR2 R\tR2 P\tR2 F1\tR1 R\tR1 P\tR1 F1\tR2 R\tR2 P\tR2 F1\n")

        for (timeline_name, gold_timeline), sys_timeline in zip(timelines, sys_timelines):
            with open(os.path.join(out_timelines_dir, timeline_name), "w") as f_tl:
                f_tl.write(str(sys_timeline))

            reference_timeline = GroundTruth([gold_timeline])
            eval_results = evaluator.evaluate_concat(sys_timeline, reference_timeline)
            rouge_1_sum += eval_results["rouge_1"]["f_score"]
            rouge_1_r_sum += eval_results["rouge_1"]["recall"]
            rouge_1_p_sum += eval_results["rouge_1"]["precision"]
            rouge_2_sum += eval_results["rouge_2"]["f_score"]
            rouge_2_r_sum += eval_results["rouge_2"]["recall"]
            rouge_2_p_sum += eval_results["rouge_2"]["precision"]


            eval_results_align = evaluator.evaluate_align_date_content_costs_many_to_one(sys_timeline, reference_timeline)
            align_rouge_1_sum += eval_results_align["rouge_1"]["f_score"]
            align_rouge_1_r_sum += eval_results_align["rouge_1"]["recall"]
            align_rouge_1_p_sum += eval_results_align["rouge_1"]["precision"]
            align_rouge_2_sum += eval_results_align["rouge_2"]["f_score"]
            align_rouge_2_r_sum += eval_results_align["rouge_2"]["recall"]
            align_rouge_2_p_sum += eval_results_align["rouge_2"]["precision"]


            print(" ".join(map(lambda x: "{}-{}-{}".format(x.year, x.month, x.day), sorted(sys_timeline))))
            print(" ".join(map(lambda x: "{}-{}-{}".format(x.year, x.month, x.day), sorted(gold_timeline))))
    
            date_recall = len(set(sys_timeline) & set(gold_timeline)) / len(gold_timeline)
            date_precision = len(set(sys_timeline) & set(gold_timeline)) / len(sys_timeline)
    
            if date_recall + date_precision > 0:
                date_f1 = 2 * (date_recall * date_precision) / (date_recall + date_precision)
            else:
                date_f1 = 0.0

            date_f1_sum += date_f1
            date_f1_r_sum += date_recall
            date_f1_p_sum += date_precision

            f_out.write("{:<16}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
                timeline_name,
                date_recall,
                date_precision,
                date_f1,
                eval_results["rouge_1"]["recall"],
                eval_results["rouge_1"]["precision"],
                eval_results["rouge_1"]["f_score"],
                eval_results["rouge_2"]["recall"],
                eval_results["rouge_2"]["precision"],
                eval_results["rouge_2"]["f_score"],

                eval_results_align["rouge_1"]["recall"],
                eval_results_align["rouge_1"]["precision"],
                eval_results_align["rouge_1"]["f_score"],
                eval_results_align["rouge_2"]["recall"],
                eval_results_align["rouge_2"]["precision"],
                eval_results_align["rouge_2"]["f_score"]

                ))

        f_out.write("{:<16}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
            "All",
            date_f1_r_sum / len(args.timelines),
            date_f1_p_sum / len(args.timelines),
            date_f1_sum / len(args.timelines),
            rouge_1_r_sum / len(args.timelines),
            rouge_1_p_sum / len(args.timelines),
            rouge_1_sum / len(args.timelines),
            rouge_2_r_sum / len(args.timelines),
            rouge_2_p_sum / len(args.timelines),
            rouge_2_sum / len(args.timelines),
            align_rouge_1_r_sum / len(args.timelines),
            align_rouge_1_p_sum / len(args.timelines),
            align_rouge_1_sum / len(args.timelines),
            align_rouge_2_r_sum / len(args.timelines),
            align_rouge_2_p_sum / len(args.timelines),
            align_rouge_2_sum / len(args.timelines)))
            #print(sys_timeline)

    print("ROUGE 1", rouge_1_sum / len(args.timelines))
    print("ROUGE 2", rouge_2_sum / len(args.timelines))
    print("Date F1", date_f1_sum / len(args.timelines))

from utils import iter_files

def cross_eval_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("corpus_def")
    parser.add_argument("config")

    args = parser.parse_args()

    corpora_and_timelines = []

    with open(args.corpus_def) as f:
        corpus_defs = json.load(f)

    for corpus_def in corpus_defs["corpora"]:
        timeline_dir = corpus_def["tl_dir"]
        corpus_pickle = corpus_def["corpus_pkl"]

        corpus = load_corpus(corpus_pickle)

        timelines = []
        for tl_fname in iter_files(timeline_dir, ".txt"):
            with open(tl_fname, encoding="latin-1") as f:
                timeline = Timeline.from_file(f)
                timelines.append((os.path.basename(tl_fname), timeline))

        corpora_and_timelines.append((corpus, timelines))

    with open(args.config) as f:
        config = json.load(f)

    tl_gen = GloballyClusteredSentenceCompressionTimelineGenerator(config)

    parameters = tl_gen.run_scoring_cv_train_mode(corpora_and_timelines)

    with open("all-parameters.pkl", "wb") as f_out:
        pickle.dump(parameters, f_out)


if __name__ == "__main__":
    cross_eval_main()
    #evaluate_tl_main()
