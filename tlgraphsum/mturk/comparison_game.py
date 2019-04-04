from tlgraphsum.cmp_tls import print_vertical_split
import sys
from tilse.data.timelines import Timeline
import itertools as it
import argparse
import datetime


def request_ranking(summary_1, summary_2):
    print(summary_1)
    print_vertical_split(summary_1, summary_2)

    while True:
        result = input()

        if result == "a":
            return -1
        elif result == "d":
            return 1
        else:
            return 0


def play_comparison_game(tl):
    all_date_pairs = it.combinations(tl.get_dates(), 2)

    for date_1, date_2 in all_date_pairs:
        ranking_result = request_ranking("\n".join(tl.dates_to_summaries[date_1]), "\n".join(tl.dates_to_summaries[date_2]))

        yield (date_1, date_2, ranking_result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("timeline")
    parser.add_argument("replay_file")
    parser.add_argument("--replay", action="store_true", default=False)

    args = parser.parse_args()

    with open(args.timeline, errors="ignore") as f:
        tl = Timeline.from_file(f)

    comparisons = play_comparison_game(tl)
    all_comparisons = []

    if not args.replay:
        with open(args.replay_file, "w") as f_out:
            for date_1, date_2, result in comparisons:
                f_out.write("{}\t{}\t{}\n".format(date_1, date_2, result))
                all_comparisons.append((date_1, date_2, result))
    else:
        with open(args.replay_file) as f:
            for line in f:
                date_1, date_2, result = line.split()
                date_1 = datetime.date(*map(int, date_1.split("-")))
                date_2 = datetime.date(*map(int, date_2.split("-")))
                result = int(result)
                all_comparisons.append((date_1, date_2, result))

    scores = {key: 0 for key in tl.get_dates()}

    for date_1, date_2, result in all_comparisons:
        if result == 0:
            scores[date_1] += 0.5
            scores[date_2] += 0.5
        elif result == -1:
            scores[date_1] += 1
        elif result == 1:
            scores[date_2] += 1

    for date, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(date, "\n", "\n".join(tl.dates_to_summaries[date]))
        print()


if __name__ == "__main__":
    main()
