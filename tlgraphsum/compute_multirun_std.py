import numpy as np
import sys
from collections import defaultdict


def main():
    with open(sys.argv[1]) as f:
        all_run_results = defaultdict(list)

        curr_topic_results = []
        curr_run_results = []

        for line in f:
            if line.startswith("---"):
                curr_topic_results.append(curr_run_results)
                curr_run_results = []
            elif line.startswith("==="):
                for run_id, tl_results in enumerate(curr_topic_results):
                    all_run_results[run_id].extend(tl_results)
                curr_topic_results = []
            else:
                elems = list(map(float, line.split()))
                curr_run_results.append(elems)

    run_scores = np.zeros(shape=(len(all_run_results), 6))
    for idx, run_results in enumerate(all_run_results.values()):
        run_scores[idx] = np.sum(run_results, axis=0) / len(all_run_results[0])
        run_results = np.array(run_results)

    #print(run_scores)
    means = np.mean(run_scores, axis=0)
    stds = np.std(run_scores, axis=0)
    print("Mean", *means, sep="\t")
    print("std", *stds, sep="\t")
    print("low", *(means - 1.97 * stds), sep="\t")
    print("high", *(means + 1.97 * stds), sep="\t")



if __name__ == "__main__":
    main()