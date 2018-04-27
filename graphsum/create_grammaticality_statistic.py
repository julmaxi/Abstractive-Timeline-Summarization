import argparse

import csv
import numpy as np

from collections import defaultdict

import krippendorff


def read_source_file(filename):
    results = []
    with open(filename) as f:
        reader = csv.reader(f)

        for item in reader:
            results.append(item)

    return results


def read_results_file(filename):
    per_annotator_results = []
    with open(filename) as f:
        reader = csv.reader(f)

        sentences = next(reader)

        for item in reader:
            per_annotator_results.append(list(map(lambda x: int(x) if len(x) != 0 else np.nan, item[2:])))
#            per_annotator_results.append(list(map(lambda x: 1 if len(x) != 0 and int(x) == 1 else 0, item[2:])))

    return per_annotator_results


def compute_alpha(item_annotations):
    matrix = []

    for batch_idx, batch in enumerate(item_annotations):
        if len(matrix) == 0:
            previous_length = 0
        else:
            previous_length = len(matrix[-1])
        for annotator_idx, annotator_data in enumerate(batch):
            matrix.append([np.nan] * previous_length + annotator_data)

    longest_row_length = max(map(len, matrix))

    for row in matrix:
        missing_nans = longest_row_length - len(row)

        if missing_nans == 0:
            continue

        row.extend([np.nan] * missing_nans)

    #print(matrix[0])
    matrix = np.array(matrix)
    #print(matrix[:2,:200])


    return krippendorff.alpha(reliability_data=matrix, level_of_measurement='interval')


def compute_statistics(all_items, item_annotations):
    item_idx_iter = iter(range(len(all_items)))

    scores = defaultdict(list)

    for block in item_annotations:
        for block_anno_idx in range(len(block[0])):
            source_info = all_items[next(item_idx_iter)]
            annotations = [annotater_items[block_anno_idx] for annotater_items in block if annotater_items[block_anno_idx] is not None]

            from collections import Counter

            val = sorted(Counter(annotations).items(), reverse=True, key=lambda x: (x[1], -x[0]))[0][0]

            scores[source_info[1]].append(val)

    for key, vals in scores.items():
        valid_vals = [v for v in vals if v is not np.nan]
        print(key, round(sum(valid_vals) / len(valid_vals), 2))

        val_freqs = Counter(vals)

        import math

        print(round(val_freqs[1] / len(valid_vals), 3), "&", round(val_freqs[2] / len(valid_vals), 3), "&", round(val_freqs[3] / len(valid_vals), 3))
        print(val_freqs)

            #print(items[0][1])
            #print(items[0][1])

#def flatten_annotations()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file")
    parser.add_argument("result_files", nargs="+")

    args = parser.parse_args()

    all_items = read_source_file(args.source_file)
    item_annotations = []

    for filename in args.result_files:
        item_annotations.append(read_results_file(filename))

    alpha = compute_alpha(item_annotations)
    print("Alpha", round(alpha, 3))

    print(compute_statistics(all_items, item_annotations))

    #flat_annotations = flatten_annotations(item_annotations)
