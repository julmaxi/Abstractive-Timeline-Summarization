from tilse.data.timelines import Timeline

from utils import iter_dirs, iter_files
import argparse
import os

from collections import defaultdict

import random

import csv


def vectorize_timelines(all_timelines, max_tls=None):
    all_tl_keys = set()

    for system_name, topic in all_timelines.items():
        for topic_name, topic_tls in topic.items():
            for tl_name in topic_tls:
                all_tl_keys.add((topic_name, tl_name))

    if max_tls is not None:
        all_tl_keys = list(all_tl_keys)
        random.shuffle(all_tl_keys)
        all_tl_keys = all_tl_keys[:max_tls]


    #print(all_tl_keys)

    all_tl_keys = sorted(all_tl_keys)

    vectorized_timelines = {}
    for system_name, topic in all_timelines.items():
        local_timelines = []
        for topic_name, tl_name in all_tl_keys:
            local_timelines.append((topic_name, tl_name, topic[topic_name][tl_name]))
        vectorized_timelines[system_name] = local_timelines

    return vectorized_timelines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("human_tl_dir")
    parser.add_argument("system_tl_dir")

    parser.add_argument("relevant_systems", nargs="+")

    parser.add_argument("outfile")

    args = parser.parse_args()
    relevant_systems = set(args.relevant_systems)

    all_relevant_timelines = defaultdict(lambda: defaultdict(dict))

    for directory in iter_dirs(args.system_tl_dir):
        system_name = os.path.basename(directory)
        for tl_dir in iter_dirs(directory):
            for tlfilename in iter_files(tl_dir, ".txt"):
                #print(system_name, relevant_systems)
                if system_name in relevant_systems:
                    with open(tlfilename) as tlfile:
                        all_relevant_timelines[system_name][os.path.basename(tl_dir)][os.path.basename(tlfilename)] = Timeline.from_file(tlfile)

    #for directory in iter_dirs(args.human_tl_dir):
    #    source_name = os.path.basename(directory)
    #    for tlfilename in iter_files(directory, ".txt"):
    #        with open(tlfilename, errors='ignore') as tlfile:
    #            all_relevant_timelines["human"][source_name][os.path.basename(tlfilename)] = Timeline.from_file(tlfile)

    vectorized_timelines = vectorize_timelines(all_relevant_timelines)

    num_samples_per_tl = 5

    all_samples = []
    for system, timelines in vectorized_timelines.items():
        system_samples = set()
        for topic_name, tl_name, timeline in timelines:
            sentences = [sent.replace("-RRB-", ")").replace("-LRB-", "(").capitalize() for date in timeline for sent in timeline[date] if len(sent) > 0]
            num_samples = min(num_samples_per_tl, len(sentences))
            system_samples.update([(sample, system, topic_name, tl_name) for sample in random.sample(sentences, num_samples)])

        if len(system_samples) < num_samples_per_tl * len(timelines):
            all_possible_system_samples = set()
            for topic_name, tl_name, timeline in timelines:
                for sample in [sent.replace("-RRB-", ")").replace("-LRB-", "(").capitalize() for date in timeline for sent in timeline[date] if len(sent) > 0]:
                    all_possible_system_samples.add((sample, system, topic_name, tl_name))

            print(len(list(all_possible_system_samples)))

            all_possible_system_samples -= system_samples

            print(len(list(all_possible_system_samples)))

            print(min(len(system_samples) - num_samples_per_tl * len(timelines), len(all_possible_system_samples)))

            system_samples.update(random.sample(list(all_possible_system_samples), min(num_samples_per_tl * len(timelines) -len(system_samples), len(all_possible_system_samples))))


        all_samples.extend(system_samples)

    random.shuffle(all_samples)

    with open(args.outfile, 'w', encoding="utf8") as csvfile:
        writer = csv.writer(csvfile)
        for sample in all_samples:
            writer.writerow(sample)
