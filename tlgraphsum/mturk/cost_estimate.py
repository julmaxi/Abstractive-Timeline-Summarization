from jinja2 import Template
import os
import sys
import argparse
from tilse.data.timelines import Timeline
from collections import defaultdict, namedtuple
import spacy
import boto3
import tlgraphsum.mturk.db as db
from xml.etree import ElementTree as ET
import time

import subprocess
import argparse
import datetime

from pathlib import PurePath, Path
import json


from tlgraphsum.utils import iter_dirs, iter_files


TURKER_PRICE_PER_TUPLE = 0.05
TURKER_PRICE_PER_READING_MINUTE = 0.10 * 2


STUDENT_PRICE_PER_TUPLE = 0.1
STUDENT_PRICE_PER_READING_MINUTE = 0.15


def flatten(l):
    result = []

    for partial_list in l:
        result.extend(partial_list)

    return result


def tokenize(s):
    return list(filter(lambda i: i.isalpha(), s.split()))


def main():
    all_timelines = []
    gold_tl_dir = "gold-timelines"
    for topic_gold_tl_dir in iter_dirs(gold_tl_dir):
        for gold_tl_fname in iter_files(topic_gold_tl_dir, ".txt"):
            with open(gold_tl_fname, errors="ignore") as f:
                tl = Timeline.from_file(f)

                print(topic_gold_tl_dir, (gold_tl_fname))

                if (os.path.split(topic_gold_tl_dir)[-1], os.path.split(gold_tl_fname)[-1]) in [
                        ("tl17-bpoil", "bbc.txt"),
                        ("crisis-syria", "bbc.txt"),
                        ("tl17-mj", "bbc.txt"),
                        ("crisis-libya", "xinhua.txt")
                    ]:
                    all_timelines.append(tl)
                    print(len(tl))

    print("Original TL-Count", len(all_timelines))
    all_timelines = list(filter(lambda i: len(i) <= 50, all_timelines))
    print("New TL-Count", len(all_timelines))

    tl_tuple_counts = [len(tl.get_dates()) * 2 for tl in all_timelines]

    print(sum(tl_tuple_counts))

    price_per_tuple = STUDENT_PRICE_PER_TUPLE
    num_annotations = 3

    tl_words = [tokenize(" ".join(summary_sents)) for tl in all_timelines for summary_sents in tl.dates_to_summaries.values()]

    tl_reading_times = [len(w) / 200 for w in tl_words]

    print("Reading time", sum(tl_reading_times))
    print("Reading cost", sum(tl_reading_times) * STUDENT_PRICE_PER_READING_MINUTE * num_annotations)

    tl_tuple_price = sum(tl_tuple_counts) * price_per_tuple * num_annotations

    print("Number of Tuples", sum(tl_tuple_counts))
    print("Annotation Time", sum(tl_tuple_counts) / 1.5)
    print("Annotation Cost", tl_tuple_price)

if __name__ == "__main__":
    main()
