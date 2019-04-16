import pickle
import sys
from collections import Counter

import matplotlib.pyplot as plt
import itertools as it
from reader import DateTag


def count_bad_pairs_in_cluster(cl):
    sent_tags = []
    for sent in cl:
        all_tags = set(sent.all_date_tags)
        all_tags.add(sent.document.dct_tag)
        sent_tags.append(all_tags)

    num_bad_pairs = 0
    num_pairs_within_a_day = 0

    for sent_tags_1, sent_tags_2 in it.combinations(sent_tags, 2):
        is_incompatible = True
        is_within_a_day = False
        for tag_1, tag_2 in it.product(sent_tags_1, sent_tags_2):
            if tag_1 in tag_2 or tag_2 in tag_1:
                is_incompatible = False
                break
            elif tag_1.dtype == DateTag.DAY and tag_2.dtype == DateTag.DAY:
                if abs((tag_1.datetime - tag_2.datetime).days) <= 2:
                    is_within_a_day = True
        if is_incompatible:
            if is_within_a_day:
                num_pairs_within_a_day += 1
            num_bad_pairs += 1
            print(sent_tags_1, sent_tags_2)

    if num_bad_pairs > 0:
        print("------------")
        for sent, tags in zip(cl, sent_tags):
            print(sent.as_tokenized_string(), tags)
        print("============")

    return num_bad_pairs, num_pairs_within_a_day


if __name__ == "__main__":
    num_bad_pairs = 0
    total_num_pairs = 0
    num_affected_clusters = 0
    num_clusters = 0
    num_clusters_larger_2 = 0
    total_num_pairs_within_a_day = 0
    for cluster_file in sys.argv[1:]:
        with open(cluster_file, "rb") as f:
            clusters = pickle.load(f)

            for cl in clusters:
                num_clusters += 1
                total_num_pairs += len(cl) * (len(cl) - 1) / 2
                if len(cl) <= 2:
                    continue

                num_clusters_larger_2 += 1
                num_cl_bad_pairs, num_cl_within_day_pairs = count_bad_pairs_in_cluster(cl)
                if num_cl_bad_pairs > 0:
                    num_affected_clusters += 1
                num_bad_pairs += num_cl_bad_pairs
                total_num_pairs_within_a_day += num_cl_within_day_pairs
    print(num_bad_pairs)
    print(total_num_pairs_within_a_day)
    print(num_bad_pairs / total_num_pairs)
    print(num_affected_clusters, num_affected_clusters / num_clusters_larger_2, num_affected_clusters / num_clusters)
