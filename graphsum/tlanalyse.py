from argparse import ArgumentParser

from utils import iter_files, iter_dirs

import os

from collections import namedtuple, defaultdict

import art

ResultEntry = namedtuple("ResultEntry", "datesel rouge_1_concat rouge_2_concat rouge_1_agree rouge_2_agree rouge_1_align rouge_2_align")
FMeasureEntry = namedtuple("FMeasureEntry", "recall precision f1")


def print_results_table(all_entries):
    all_names = all_entries.keys()

    longest_name_len = max(map(len, all_names))

    val_headers = "Date F1", "R1 concat", "R2 concat", "R1 agree", "R2 agree", "R1 align", "R2 align"

    print("{}\t{}".format("System".ljust(longest_name_len), "\t".join(val_headers)))

    for sys_name, entry in sorted(all_entries.items(), key=lambda i: i[1].rouge_2_concat.f1, reverse=True):
        cells = [sys_name.ljust(longest_name_len)]

        for val, header in zip((entry.datesel.f1, entry.rouge_1_concat.f1, entry.rouge_2_concat.f1, entry.rouge_1_agree.f1, entry.rouge_2_agree.f1, entry.rouge_1_align.f1, entry.rouge_2_align.f1), val_headers):
            cells.append("{:.3f}".format(val).ljust(len(header)))

        print("\t".join(cells))



def analyze_main():
    results_basedir = "evaluation_results"

    all_entries = {}
    tl17_entries = {}
    crisis_entries = {}

    for system_dir in iter_dirs(results_basedir):
        entry, tl17_entry, crisis_entry = analyze_system_results_dir(system_dir)
        if entry is not None:
            all_entries[os.path.basename(system_dir)] = entry

        if tl17_entry is not None:
            tl17_entries[os.path.basename(system_dir)] = tl17_entry
        if crisis_entry is not None:
            crisis_entries[os.path.basename(system_dir)] = crisis_entry

    print_results_table(all_entries)
    print("\n===== TL17 =====")
    print_results_table(tl17_entries)
    print("\n===== Crisis =====")
    print_results_table(crisis_entries)


    #all_entries = sorted(all_entries.items())
#
    #print("System\t\t", "\t".join(("Date F1", "R1 concat", "R2 concat", "R1 align", "R2 align")))
    #for sys_name, entry in all_entries:
    #    print("\t".join(map(str, (sys_name, entry.datesel.f1, entry.rouge_1_concat.f1, entry.rouge_2_concat.f1, entry.rouge_1_align.f1, entry.rouge_2_align.f1))))


def analyze_system_results_dir(results_dir):
    all_results = {}

    tl17_results = {}
    crisis_results = {}

    relevant_files = list(iter_files(results_dir, ".txt"))

    if len(relevant_files) == 0:
        return None, None, None

    for results_file in relevant_files:
        topic_result = {}

        with open(results_file) as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue

                components = line.strip().split()

                if components[0].lower() == "all":
                    continue

                tl_data = []

                for range_start in range(1, len(components), 3):
                    tl_data.append(FMeasureEntry(*map(float, components[range_start:range_start + 3])))

                entry = ResultEntry(*tl_data)

                topic_result[components[0]] = entry

        all_results[os.path.basename(results_file)] = topic_result

        if "crisis" in os.path.basename(results_file):
            crisis_results[os.path.basename(results_file)] = topic_result
        elif "tl17" in os.path.basename(results_file):
            tl17_results[os.path.basename(results_file)] = topic_result
        all_results[os.path.basename(results_file)] = topic_result

    macro_average_entry = compute_macro_averages(all_results)
    if len(tl17_results) > 0:
        tl17_macro_average_entry = compute_macro_averages(tl17_results)
    else:
        tl17_macro_average_entry = None

    if len(crisis_results) > 0:
        crisis_macro_average_entry = compute_macro_averages(crisis_results)
    else:
        crisis_macro_average_entry = None

    return macro_average_entry, tl17_macro_average_entry, crisis_macro_average_entry


def compute_macro_averages(topic_results):
    global_average_result_entries = defaultdict(lambda: [0., 0., 0.])

    for tl_results in topic_results.values():
        topic_average_result_entries = defaultdict(lambda: [0., 0., 0.])
        for entry in tl_results.values():
            for metric in "datesel rouge_1_concat rouge_2_concat rouge_1_agree rouge_2_agree rouge_1_align rouge_2_align".split():
                topic_average_result_entries[metric][0] += getattr(entry, metric)[0]
                topic_average_result_entries[metric][1] += getattr(entry, metric)[1]
                topic_average_result_entries[metric][2] += getattr(entry, metric)[2]

        for metric in "datesel rouge_1_concat rouge_2_concat rouge_1_agree rouge_2_agree rouge_1_align rouge_2_align".split():
            global_average_result_entries[metric][0] += topic_average_result_entries[metric][0] / len(tl_results)
            global_average_result_entries[metric][1] += topic_average_result_entries[metric][1] / len(tl_results)
            global_average_result_entries[metric][2] += topic_average_result_entries[metric][2] / len(tl_results)

    entry_params = []
    for metric in "datesel rouge_1_concat rouge_2_concat rouge_1_agree rouge_2_agree rouge_1_align rouge_2_align".split():
        entry_params.append(FMeasureEntry(*map(lambda v: v / len(topic_results), global_average_result_entries[metric])))

    return ResultEntry(*entry_params)


if __name__ == "__main__":
    analyze_main()
