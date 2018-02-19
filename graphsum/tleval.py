from tilse.data import timelines
from tilse.evaluation import rouge
import argparse

from tlsum import create_timeline_sentence_level, TimelineParameters

import pickle


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


def evaluate_tl_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", dest="timelines", nargs="+")
    parser.add_argument("corpus_pickle")

    args = parser.parse_args()

    evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])

    rouge_1_sum = 0
    rouge_2_sum = 0

    with open(args.corpus_pickle, "rb") as f:
        corpus = pickle.load(f)

    for tl_fname in args.timelines:
        with open(tl_fname) as f:
            timeline = timelines.Timeline.from_file(f)
        tl = create_timeline_sentence_level(corpus, determine_tl_parameters(timeline))
        reference_timeline = timelines.GroundTruth([timeline])
        eval_results = evaluator.evaluate_concat(tl, reference_timeline)
        rouge_1_sum += eval_results["rouge_1"]["f_score"]
        rouge_2_sum += eval_results["rouge_2"]["f_score"]

    print("ROUGE 1", rouge_1_sum / len(args.timelines))
    print("ROUGE 2", rouge_2_sum / len(args.timelines))


if __name__ == "__main__":
    evaluate_tl_main()
