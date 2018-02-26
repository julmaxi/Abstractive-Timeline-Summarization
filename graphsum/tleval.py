from tilse.data.timelines import Timeline, GroundTruth
from tilse.evaluation import rouge
import argparse

from tlsum import create_timeline_sentence_level, TimelineParameters, APClusteringTimelineGenerator

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
    date_f1_sum = 0

    with open(args.corpus_pickle, "rb") as f:
        corpus = pickle.load(f)

    timelines = []

    for tl_fname in args.timelines:
        with open(tl_fname, encoding="latin-1") as f:
            timeline = Timeline.from_file(f)
            timelines.append(timeline)

    tl_gen = APClusteringTimelineGenerator(True)
    sys_timelines = tl_gen.generate_timelines(corpus, [determine_tl_parameters(tl) for tl in timelines])
    for gold_timeline, sys_timeline in zip(timelines, sys_timelines):
        reference_timeline = GroundTruth([gold_timeline])
        eval_results = evaluator.evaluate_concat(sys_timeline, reference_timeline)
        rouge_1_sum += eval_results["rouge_1"]["f_score"]
        rouge_2_sum += eval_results["rouge_2"]["f_score"]

        print(" ".join(map(lambda x: "{}-{}-{}".format(x.year, x.month, x.day), sorted(sys_timeline))))
        print(" ".join(map(lambda x: "{}-{}-{}".format(x.year, x.month, x.day), sorted(gold_timeline))))

        date_recall = len(set(sys_timeline) & set(gold_timeline)) / len(gold_timeline)
        date_precision = len(set(sys_timeline) & set(gold_timeline)) / len(sys_timeline)

        if date_recall + date_precision > 0:
            date_f1_sum += 2 * (date_recall * date_precision) / (date_recall + date_precision)

        #print(sys_timeline)

    print("ROUGE 1", rouge_1_sum / len(args.timelines))
    print("ROUGE 2", rouge_2_sum / len(args.timelines))
    print("Date F1", date_f1_sum / len(args.timelines))


if __name__ == "__main__":
    evaluate_tl_main()
