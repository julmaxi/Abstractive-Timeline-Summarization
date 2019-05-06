from jinja2 import Template
import os
import sys
import argparse
from tilse.data.timelines import Timeline
from collections import defaultdict, namedtuple, Counter
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
import numpy as np
import scipy.stats
import krippendorff as k

from tlgraphsum.mturk.post_bw_task import MTurkClient, str_to_tl_date

import itertools as it


def compute_bradley_terry(items, all_matches):
    scores = np.full(len(items), 1.0 / items)

    items_to_indices = {item: idx for idx, item in items}

    number_of_wins = np.zeros(len(items))
    number_of_matches = np.zeros((len(items), len(items)))

    for win_item, loose_item in all_matches:
        win_idx = items_to_indices[win_item]
        loose_idx = items_to_indices[loose_item]
        number_of_wins[win_idx] += 1
        number_of_matches[win_idx, loose_idx] += 1
        number_of_matches[loose_idx, win_idx] += 1

    while True:
        new_scores = np.full(len(items), 0.0)

        for idx, item in enumerate(items):
            for other_idx, other_item in enumerate(items):
                new_scores[idx] += number_of_matches[idx, other_idx] / (scores[idx] + scores[other_idx])

        new_scores **= -1
        new_scores /= number_of_wins

        print(new_scores)


ANNOTATORS = {
    "A1X2R1CPERG5PL": "Julius",
    "A2HTAOJPX5UX47": "Sibylla",
    "A2JNNFC987NAB6": "Katja"
}


def main():
    info_path = Path("./hitinfo")

    client = MTurkClient("Timeline Importance Annotation VIII")
    client2 = MTurkClient("Timeline Importance Annotation VIII (K)")

    hit_ids = client.list_reviewable_hit_ids_and_annotations()
    hit_ids.extend(client2.list_reviewable_hit_ids_and_annotations())

    date_scores = defaultdict(lambda: defaultdict(Counter))

    binary_relations_per_annotator = defaultdict(lambda: defaultdict(list))
    all_tasks_items = defaultdict(set)

    for hit_id, info in hit_ids:
        results = client.load_answers_for_hit(hit_id)
        task_name, hit_info = info.split(":")
        tl_topic, tl_name = hit_info.split("__")

        info_path = Path("./hitinfo") / hit_id

        info_text = info_path.read_text()
        tuples = []
        for line in info_text.split("\n")[1:]:
            tuples.append(line.split())

            all_tasks_items[tl_topic, tl_name].update(line.split())

        for assignments in results:
            for worker_id, assignment_answer in assignments:
                for key, answer in assignment_answer.items():
                    _, row_idx, answer_type = key.split("-")
                    date_scores[worker_id][tl_topic, tl_name][str_to_tl_date(answer)] += (1 if answer_type == "best" else -1)

                    if answer_type == "best":
                        binary_relations_per_annotator[tl_topic, tl_name][worker_id].extend(
                            (answer, t) for t in tuples[int(row_idx)] if t != answer
                        )
                    else:
                        binary_relations_per_annotator[tl_topic, tl_name][worker_id].extend(
                            (t, answer) for t in tuples[int(row_idx)] if t != answer
                        )

    for task_id, task in binary_relations_per_annotator.items():
        print(task_id)
        all_headers = all_tasks_items[task_id]
        pairwise_contradictions = []
        for annotator, binary_relations in binary_relations_per_annotator[task_id].items():
            for item_1, item_2 in it.combinations(all_headers, 2):
                if (item_1, item_2) in binary_relations and (item_2, item_1) in binary_relations:
                    pairwise_contradictions.append((item_1, item_2))

            triple_contradictions = []

            for items in it.combinations(all_headers, 3):
                for item_1, item_2, item_3 in it.permutations(items):
                    if (item_1, item_2) in binary_relations and (item_2, item_3) in binary_relations and (item_3, item_1) in binary_relations:
                        triple_contradictions.append((item_1, item_3))
                        break

            print(annotator, len(pairwise_contradictions), len(triple_contradictions))


    compute_reliability(date_scores)


def compute_reliability(date_scores):
    tl_base_path = Path("./gold-timelines")
    all_annotators = sorted(date_scores)

    all_topics = set()

    for scores in date_scores.values():
        for topic, tl_name in scores:
            all_topics.add((topic, tl_name))

    for topic, tl_name in all_topics:
        with open(tl_base_path / topic / (tl_name + ".txt"), errors="ignore") as f:
            tl = Timeline.from_file(f)

            score_matrix = np.zeros((len(date_scores), len(tl.get_dates())))

            all_dates = sorted(tl.get_dates())

            for annotator_idx, annotator in enumerate(all_annotators):
                annotator_tl_scores = date_scores[annotator][(topic, tl_name)]

                sorted_dates = sorted(all_dates, key=lambda date: annotator_tl_scores.get(date, 0), reverse=True)
                new_annotator_tl_scores = {}
                curr_idx = 0
                prev_score = None
                for date in sorted_dates:
                    score = annotator_tl_scores.get(date, 0)
                    if prev_score is None or prev_score != score:
                        curr_idx += 1
                        prev_score = score
                    new_annotator_tl_scores[date] = curr_idx

                for date_idx, date in enumerate(all_dates):
                    score_matrix[annotator_idx, date_idx] = new_annotator_tl_scores.get(date, 0)

        print(topic, tl_name, k.alpha(score_matrix, level_of_measurement="ordinal"))

        for annotator_1_idx, annotator_2_idx in it.combinations(range(len(all_annotators)), 2):
            annotator_1 = ANNOTATORS[all_annotators[annotator_1_idx]]
            annotator_2 = ANNOTATORS[all_annotators[annotator_2_idx]]

            annotator_rows = score_matrix[[annotator_1_idx,annotator_2_idx]]

            print(annotator_1, annotator_2, k.alpha(annotator_rows, level_of_measurement="interval"))


def main_twotasks():
    annotations_j = retrieve_hit_ranking("Timeline Importance Annotation V (J)")
    annotations_k = retrieve_hit_ranking("Timeline Importance Annotation VI (K)")

    assert list(annotations_j) == list(annotations_k)

    tl_base_path = Path("./gold-timelines")

    for topic, tl_name in annotations_j:
        with open(tl_base_path / topic / (tl_name + ".txt"), errors="ignore") as f:
            tl = Timeline.from_file(f)

        all_dates = tl.get_dates()

        scores_j = annotations_j[topic, tl_name]
        scores_k = annotations_k[topic, tl_name]

        for date in all_dates:
            if date not in scores_j:
                scores_j[date] = 0
            if date not in scores_k:
                scores_k[date] = 0

        obs_j = np.zeros(len(tl.get_dates()))
        obs_k = np.zeros(len(tl.get_dates()))
        
        top_5_dates_j = [i[0] for i in sorted(scores_j.items(), key=lambda i: i[1])[:10]]
        top_5_dates_k = [i[0] for i in sorted(scores_k.items(), key=lambda i: i[1])[:10]]

        print(top_5_dates_j)
        print(top_5_dates_k)
        print(len(set(top_5_dates_j).intersection(set(top_5_dates_k))) / len(top_5_dates_j))

        ranking_j = map(lambda x: x[0], sorted(scores_j.items(), key=lambda x: x[1], reverse=True))
        ranking_k = map(lambda x: x[0], sorted(scores_k.items(), key=lambda x: x[1], reverse=True))

        ranks_j = dict(map(lambda x: tuple(reversed(x)), enumerate(ranking_j)))
        ranks_k = dict(map(lambda x: tuple(reversed(x)), enumerate(ranking_k)))

        print(ranks_j)

        for idx, date in enumerate(all_dates):
            score_j = scores_j[date]
            score_k = scores_k[date]

            obs_j[idx] = score_j
            obs_k[idx] = score_k

            print(ranks_j[date] + 1, ranks_k[date] + 1)
            print(score_j, score_k)
            print(date)
            print(tl.dates_to_summaries[date])

        print(k.alpha(np.stack([obs_j, obs_k]), level_of_measurement="interval"))

        print(scipy.stats.kendalltau(obs_j, obs_k))


def retrieve_hit_ranking(hit_task_name):
    client = MTurkClient(hit_task_name)

    hit_ids = client.list_reviewable_hit_ids_and_annotations()

    date_scores = defaultdict(Counter)

    for hit_id, info in hit_ids:
        results = client.load_answers_for_hit(hit_id)
        task_name, hit_info = info.split(":")
        tl_topic, tl_name = hit_info.split("__")

        for assignment in results:
            for worker_id, assignment_answer in assignment:
                for key, answer in assignment_answer.items():
                    _, row_idx, answer_type = key.split("-")
                    date_scores[tl_topic, tl_name][str_to_tl_date(answer)] += 1 if answer_type == "best" else -1

                    if "K" not in hit_task_name:
                        print(answer, answer_type)

    return date_scores

    tl_base_path = Path("./gold-timelines")

    timelines_and_scores = {}

    for (topic, tl_name), results in date_scores.items():
        with open(tl_base_path / topic / (tl_name + ".txt"), errors="ignore") as f:
            tl = Timeline.from_file(f)

        timelines_and_scores[topic, tl_name] = (results, tl)

        #summaries = sorted(tl.dates_to_summaries.items(), key=lambda it: results[it[0]], reverse=True)

        #if (info_path / hit_id).is_file():
        #    pass

    return timelines_and_scores



if __name__ == "__main__":
    main()
