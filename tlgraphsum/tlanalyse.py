from argparse import ArgumentParser

from utils import iter_files, iter_dirs

import os
import re

from collections import namedtuple, defaultdict

import art.scores
import art.significance_tests
import art.aggregators
from tleval import load_corpus
from tilse.data.timelines import Timeline

metrics = "datesel rouge_1_concat rouge_2_concat rouge_1_agree rouge_2_agree rouge_1_align rouge_2_align"

ResultEntry = namedtuple("ResultEntry", metrics + " num_tl num_sent num_copied")
FMeasureEntry = namedtuple("FMeasureEntry", "recall precision f1")


def print_results_table(all_entries, compute_copy_rate):
    all_names = all_entries.keys()

    if len(all_names) == 0:
        return

    longest_name_len = max(map(len, all_names))

    val_headers = "Date F1", "R1 concat", "R2 concat", "R1 agree", "R2 agree", "R1 align", "R2 align", "#TL"

    if compute_copy_rate:
        val_headers = val_headers + ("%copy",)

    print("{}\t{}".format("System".ljust(longest_name_len), "\t".join(val_headers)))

    for sys_name, entry in sorted(all_entries.items(), key=lambda i: i[1].rouge_1_align.f1, reverse=True):

        cells = [sys_name.ljust(longest_name_len)]

        for val, header in zip((entry.datesel.f1, entry.rouge_1_concat.f1, entry.rouge_2_concat.f1, entry.rouge_1_agree.f1, entry.rouge_2_agree.f1, entry.rouge_1_align.f1, entry.rouge_2_align.f1, entry.num_tl), val_headers):
            cells.append("{:.3f}".format(val).ljust(len(header)))

        if compute_copy_rate:
            cells.append("{:.3f}".format(entry.num_copied / entry.num_sent))

        print("\t".join(cells))


def check_significance(all_entries, key_1, key_2):
    all_timeline_names = sorted(set([(topic_key, tl_key) for topic_key, tls in all_entries[key_1].items() for tl_key in tls.keys()]))

    all_scores = defaultdict(lambda: [[], []])
    for topic_name, tl_name in all_timeline_names:
        for score in metrics.split():
            if len(all_entries[key_1][topic_name]) == len(all_entries[key_1][topic_name]) == 1: # Stupid workaround for incompatible timeline names in tl17-mj
                score_1 = art.scores.Score([(getattr(list(all_entries[key_1][topic_name].values())[0], score)).f1])
                score_2 = art.scores.Score([(getattr(list(all_entries[key_2][topic_name].values())[0], score)).f1])
            else:
                score_1 = art.scores.Score([getattr(all_entries[key_1][topic_name][tl_name], score).f1])
                score_2 = art.scores.Score([getattr(all_entries[key_2][topic_name][tl_name], score).f1])

            all_scores[score][0].append(score_1)
            all_scores[score][1].append(score_2)

    results = {}

    for score_key, (scores_1, scores_2) in all_scores.items():
        level = art.significance_tests.ApproximateRandomizationTest(
            art.scores.Scores(scores_1),
            art.scores.Scores(scores_2),
            art.aggregators.average).run()
        results[score_key] = level

    return results


def analyze_main():
    parser = ArgumentParser()
    parser.add_argument("-c", dest="compute_copy_rate", action="store_true", default=False)
    parser.add_argument("-f", dest="system_filters", nargs="+")
    parser.add_argument("-b", dest="results_basedir", default=".")

    args = parser.parse_args()

    results_basedir = os.path.join(args.results_basedir, "evaluation_results")

    all_entries = {}
    tl17_entries = {}
    crisis_entries = {}

    all_tl17_results = {}
    all_crisis_results = {}

    if args.system_filters:
        prefixes = [re.compile(expr) for expr in args.system_filters]
    else:
        prefixes = None

    for system_dir in iter_dirs(results_basedir):
        system_name = os.path.split(system_dir.rstrip("/"))[1]

        if prefixes is not None and not any(p.match(system_name) for p in prefixes):
            continue

        entry, tl17_entry, crisis_entry, tl17_results, crisis_results = analyze_system_results_dir(system_dir, compute_copy_rate=args.compute_copy_rate, basepath=args.results_basedir)

        if entry is not None:
            all_entries[os.path.basename(system_dir)] = entry

        if tl17_entry is not None:
            tl17_entries[os.path.basename(system_dir)] = tl17_entry
        if crisis_entry is not None:
            crisis_entries[os.path.basename(system_dir)] = crisis_entry

        all_tl17_results[os.path.basename(system_dir)] = tl17_results
        all_crisis_results[os.path.basename(system_dir)] = crisis_results

   #baseline_sig = check_significance(all_tl17_results, "agglo-abstractive-temptr-dateref-clsize-path.json", "baseline.json")
   #full_system_sig = check_significance(all_tl17_results, "ap-abstractive-temptr-dateref-clsize-path.json+tok", "agglo-abstractive-temptr-dateref-clsize-path.json")

    #print_results_table(all_entries, compute_copy_rate=args.compute_copy_rate, restricted_to_prefixes=prefixes)
    print("\n===== TL17 =====")
    print_results_table(tl17_entries, compute_copy_rate=args.compute_copy_rate)
    print("\n===== Crisis =====")
    print_results_table(crisis_entries, compute_copy_rate=args.compute_copy_rate)

    if False:
        for (entries, results) in ((crisis_entries, all_crisis_results), (tl17_entries, all_tl17_results)):
            for key, entry in entries.items():
                if "full-no" in key and "+sent" in key:
                    datesel_diff = (entry.datesel.f1 - entries["ap-abstractive-datetr-dateref-clsize-path-depfiltered-greedy-redundancy.json+sent"].datesel.f1)
                    rouge_1_diff = (entry.rouge_1_align.f1 - entries["ap-abstractive-datetr-dateref-clsize-path-depfiltered-greedy-redundancy.json+sent"].rouge_1_align.f1)
                    rouge_2_diff = (entry.rouge_2_align.f1 - entries["ap-abstractive-datetr-dateref-clsize-path-depfiltered-greedy-redundancy.json+sent"].rouge_2_align.f1)

                    sig_dict = (check_significance(results, "ap-abstractive-datetr-dateref-clsize-path-depfiltered-greedy-redundancy.json+sent", key))
                    print(sig_dict)
                    cell_str = "{}&\t{:.3f}\t&\t{:.3f}&\t&\t{:.3f}\\\\".format(key, datesel_diff, rouge_1_diff, rouge_2_diff)
                    print(cell_str)

    #print()
    #gen_latex_table_oracle(tl17_entries, crisis_entries, all_tl17_results, all_crisis_results)
    #print()
    #gen_latex_table_oracle(tl17_entries, crisis_entries, all_tl17_results, all_crisis_results, use_tok=True)
    #print()

    #print()
    #gen_latex_table_sent_features(tl17_entries, crisis_entries, all_tl17_results, all_crisis_results)
    #print()

    #gen_latex_table_sent(tl17_entries, crisis_entries, all_tl17_results, all_crisis_results)
    print()
    #print()
    #gen_latex_table_simple([
    #    "ap-abstractive-oracle-depfiltered-greedy.json+sent",
    #    "extractive-oracle-greedy.json+sent",
    #], ["AP Oracle", "Extractive Oracle"], [
    #    ("ap-abstractive-oracle-depfiltered-greedy.json+sent", "extractive-oracle-greedy.json+sent", "*")
    #], tl17_entries, crisis_entries, all_tl17_results, all_crisis_results)

    #gen_latex_table_simple([
    #    "extractive-oracle-greedy.json+sent",
    #    "extractive-dctdating-oracle-greedy.json+sent"
    #], ["TimeEx", "DCT"], [
    #    ("extractive-oracle-greedy.json+sent", "extractive-dctdating-oracle-greedy.json+sent", "*")
    #], tl17_entries, crisis_entries, all_tl17_results, all_crisis_results)

    print()
    print()

    gen_latex_table_simple([
        ("chieu", "Chieu"),
        ("nn-lr-pred+none", "Neural"),
        ("submod", "Submod (s)"),
        ("extractive-baseline.json+sent", "Extr. (s)"),
        #"ap-extractive-baseline-redundancy.json+sent",
        ("ap-abstractive-datetr-dateref-clsize-path-depfiltered-greedy-redundancy.json+sent", "Abstractive (s)"),
        ("extractive-oracle-greedy.json+sent", "Extr. Oracle (s)"),
        ("ap-abstractive-oracle-depfiltered-greedy-redundancy.json+sent", "Abs. Oracle (s)"),
        ("submod-tok", "Sumod (t)"),
        ("extractive-baseline.json+tok", "Extr. (t)"),
        #"ap-extractive-baseline-redundancy.json+tok",
        ("ap-abstractive-datetr-dateref-clsize-path-depfiltered-greedy-redundancy.json+tok", "Abstractive (t)"),
        ("extractive-oracle-greedy.json+tok", "Ext. Oracle (t)"),
        ("ap-abstractive-oracle-depfiltered-greedy-redundancy.json+tok", "Abs. Oracle (t)")
    ],
    [
        ("chieu", "ap-abstractive-datetr-dateref-clsize-path-depfiltered-greedy-redundancy.json+sent", "1"),
        ("chieu", "ap-abstractive-datetr-dateref-clsize-path-depfiltered-greedy-redundancy.json+tok", "a"),
        ("nn-lr-pred+none", "ap-abstractive-datetr-dateref-clsize-path-depfiltered-greedy-redundancy.json+sent", "2"),
        ("nn-lr-pred+none", "ap-abstractive-datetr-dateref-clsize-path-depfiltered-greedy-redundancy.json+tok", "b"),
        ("submod", "ap-abstractive-datetr-dateref-clsize-path-depfiltered-greedy-redundancy.json+sent", "3"),
        ("submod-tok", "ap-abstractive-datetr-dateref-clsize-path-depfiltered-greedy-redundancy.json+tok", "c"),
        ("extractive-baseline.json+sent", "ap-abstractive-datetr-dateref-clsize-path-depfiltered-greedy-redundancy.json+sent", "*"),
        ("extractive-baseline.json+tok", "ap-abstractive-datetr-dateref-clsize-path-depfiltered-greedy-redundancy.json+tok", "*"),
        ("ap-abstractive-oracle-depfiltered-greedy-redundancy.json+tok", "extractive-oracle-greedy.json+tok", "*"),
        ("ap-abstractive-oracle-depfiltered-greedy-redundancy.json+sent", "extractive-oracle-greedy.json+sent", "*")
    ], tl17_entries, crisis_entries, all_tl17_results, all_crisis_results)


    significance_pairs = [
        #('apuntangle-abstractive-oracle-depfiltered-greedy.json', 'extractive-oracle-greedy.json'),
        #('ap-abstractive-oracle-depfiltered-greedy.json', 'extractive-oracle-greedy.json'),
        ('ap-abstractive-datetr-dateref-clsize-path-depfiltered-greedy.json', 'martschat.json'),
        ('ap-extractive-baseline-noscores.json+tok', 'extractive-baseline.json'),
       #('ap-abstractive-oracle-depfiltered-greedy.json', 'extractive-oracle-greedy.json'),
       #('ap-abstractive-temptr-dateref-clsize-path-depfiltered-greedy.json', 'extractive-baseline.json'),
       #('ap-abstractive-temptr-dateref-clsize-path-depfiltered.json', 'extractive-baseline.json'),

       #('ap-abstractive-oracle-depfiltered-greedy.json', 'extractive-oracle-nosubmod.json'),
       #('ap-abstractive-oracle-depfiltered-greedy.json', 'extractive-oracle.json')
       # ("ap-abstractive-temptr-dateref-clsize-path.json", "ap-abstractive-datetr-dateref-path.json")
       # ("extractive-oracle.json", "ap-abstractive-oracle.json"),
       # ("agglo-abstractive-oracle.json", "ap-abstractive-oracle.json"),
       # ("baseline-oracle.json", "agglo-abstractive-oracle.json"),
    ]

    for system_config in "tok", "sent":
        for sys_key_1, sys_key_2 in significance_pairs:
            key_1 = sys_key_1 + "+" + system_config
            key_2 = sys_key_2 + "+" + system_config
            for corpus in ("tl17", "crisis"):
                if corpus == "tl17":
                    all_results = all_tl17_results
                else:
                    all_results = all_crisis_results

                results = check_significance(all_results, key_1, key_2)
                if results == {}:
                    continue

                all_results = []
                for metric in metrics.split():
                    result = results[metric]
                    if result <= 0.05 or result >= 0.95:
                        all_results.append("X")
                    else:
                        all_results.append("-")

                print(corpus, key_1, key_2, " ".join(all_results))

    cross_constr_significance_checks = ["extractive-oracle.json", "agglo-abstractive-oracle.json", "ap-abstractive-oracle.json"]

    for key in cross_constr_significance_checks:
        key_1 = key + "+tok"
        key_2 = key + "+sent"
        for corpus in ("tl17", "crisis"):
            if corpus == "tl17":
                all_results = all_tl17_results
            else:
                all_results = all_crisis_results
            results = check_significance(all_results, key_1, key_2)

            all_results = []
            for metric in metrics.split():
                result = results[metric]
                if result <= 0.05:
                    all_results.append("X")
                else:
                    all_results.append("-")

            print(corpus, key_1, key_2, " ".join(all_results))


    #all_entries = sorted(all_entries.items())
#
    #print("System\t\t", "\t".join(("Date F1", "R1 concat", "R2 concat", "R1 align", "R2 align")))
    #for sys_name, entry in all_entries:
    #    print("\t".join(map(str, (sys_name, entry.datesel.f1, entry.rouge_1_concat.f1, entry.rouge_2_concat.f1, entry.rouge_1_align.f1, entry.rouge_2_align.f1))))


def count_tl_copied_sentences(corpus_sentences, tl):
    num_copied = 0
    for date in tl:
        for sent in tl[date]:
            sent = sent.lower()
            if tuple(sent.split()) in corpus_sentences:
                num_copied += 1

    return num_copied



class CachedCorpusReader:
    loaded_corpus = None
    loaded_corpus_name = None

    @classmethod
    def load_corpus(cls, name):
        if name != CachedCorpusReader.loaded_corpus_name:
            name = name.split("+")[0] + ".pkl"
            CachedCorpusReader.loaded_corpus = load_corpus(name, filter_blacklist=False)
            CachedCorpusReader.loaded_corpus_name = name
        corpus = CachedCorpusReader.loaded_corpus
        return corpus


def analyse_results_file(results_file, compute_copy_rate=False, basepath="."):
    system_path, topic_fname = os.path.split(results_file)
    topic_name = topic_fname.rsplit(".", 1)[0]
    _, system_name = os.path.split(system_path)

    sys_tl_path = os.path.join(basepath, "system_timelines", system_name, topic_name)
    corpus_name = os.path.join("corpora", topic_name + ".pkl")

    if compute_copy_rate:
        corpus = CachedCorpusReader.load_corpus(corpus_name)

    with open(results_file) as f:
        topic_result = {}
        for idx, line in enumerate(f):
            if idx == 0:
                continue

            components = line.strip().split()

            if components[0].lower() == "all":
                continue

            tl_name = components[0]
            tl_fname = os.path.join(sys_tl_path, tl_name)
            num_copied = None
            total_sent_count = 0

            if compute_copy_rate:
                with open(tl_fname) as f_tl:
                    tl = Timeline.from_file(f_tl)
                    num_copied = count_tl_copied_sentences(set(map(lambda s: tuple(s.as_token_attr_sequence("form_lowercase")), corpus.sentences)),tl)
                    total_sent_count = tl.get_number_of_sentences()

            tl_data = []

            for range_start in range(1, len(components), 3):
                tl_data.append(FMeasureEntry(*map(float, components[range_start:range_start + 3])))

            tl_data.append(1)  # Number of timelines for this entry
            tl_data.append(total_sent_count)
            tl_data.append(num_copied)

            entry = ResultEntry(*tl_data)

            topic_result[components[0]] = entry
    return topic_result


def analyze_system_results_dir(results_dir, compute_copy_rate=False, macro_average=False, basepath="."):
    all_results = {}

    tl17_results = {}
    crisis_results = {}

    relevant_files = list(iter_files(results_dir, ".txt"))

    if len(relevant_files) == 0:
        return None, None, None, None, None

    for results_file in relevant_files:
        if "+filtered" not in results_file:
            print(results_file)
            continue
        #if "libya" in results_file and "crisis" in results_file:
        #    print(results_file)
        #    continue

        topic_result = analyse_results_file(results_file, compute_copy_rate=compute_copy_rate, basepath=basepath)

        all_results[os.path.basename(results_file)] = topic_result

        if "crisis" in os.path.basename(results_file):
            crisis_results[os.path.basename(results_file)] = topic_result
        elif "tl17" in os.path.basename(results_file):
            tl17_results[os.path.basename(results_file)] = topic_result
        else:
            print("Can't match results file to corpus", results_file)
        all_results[os.path.basename(results_file)] = topic_result

    if macro_average:
        macro_average_entry = compute_macro_averages(all_results)
    else:
        macro_average_entry = compute_micro_averages(all_results)
    if len(tl17_results) > 0:
        if macro_average:
            tl17_macro_average_entry = compute_macro_averages(tl17_results)
        else:
            tl17_macro_average_entry = compute_micro_averages(tl17_results)
    else:
        tl17_macro_average_entry = None

    if len(crisis_results) > 0:
        if macro_average:
            crisis_macro_average_entry = compute_macro_averages(crisis_results)
        else:
            crisis_macro_average_entry = compute_micro_averages(crisis_results)
    else:
        crisis_macro_average_entry = None

    return macro_average_entry, tl17_macro_average_entry, crisis_macro_average_entry, tl17_results, crisis_results



def compute_micro_averages(topic_results):
    global_average_result_entries = defaultdict(lambda: [0., 0., 0.])

    num_tl = 0
    sum_copied = None
    sum_total = 0

    for topic_name, tl_results in topic_results.items():
        topic_average_result_entries = defaultdict(lambda: [0., 0., 0.])
        for entry in tl_results.values():
            for metric in metrics.split():
                global_average_result_entries[metric][0] += getattr(entry, metric)[0]
                global_average_result_entries[metric][1] += getattr(entry, metric)[1]
                global_average_result_entries[metric][2] += getattr(entry, metric)[2]

            num_tl += 1

            sum_total += entry.num_sent
            if entry.num_copied is not None:
                if sum_copied is not None:
                    sum_copied += entry.num_copied
                else:
                    sum_copied = entry.num_copied

    entry_params = []
    for metric in metrics.split():
        entry_params.append(FMeasureEntry(*map(lambda v: v / num_tl, global_average_result_entries[metric])))

    entry_params.append(num_tl)
    entry_params.append(sum_total)
    entry_params.append(sum_copied)

    return ResultEntry(*entry_params)


def compute_macro_averages(topic_results):
    global_average_result_entries = defaultdict(lambda: [0., 0., 0.])

    for topic_name, tl_results in topic_results.items():
        topic_average_result_entries = defaultdict(lambda: [0., 0., 0.])
        for entry in tl_results.values():
            for metric in metrics.split():
                topic_average_result_entries[metric][0] += getattr(entry, metric)[0]
                topic_average_result_entries[metric][1] += getattr(entry, metric)[1]
                topic_average_result_entries[metric][2] += getattr(entry, metric)[2]

        for metric in metrics.split():
            global_average_result_entries[metric][0] += topic_average_result_entries[metric][0] / len(tl_results)
            global_average_result_entries[metric][1] += topic_average_result_entries[metric][1] / len(tl_results)
            global_average_result_entries[metric][2] += topic_average_result_entries[metric][2] / len(tl_results)

    entry_params = []
    for metric in metrics.split():
        entry_params.append(FMeasureEntry(*map(lambda v: v / len(topic_results), global_average_result_entries[metric])))

    return ResultEntry(*entry_params)

import numpy as np

def gen_latex_table_oracle(tl17_entries, crisis_entries, all_tl17_results=None, all_crisis_results=None, use_tok=False):
    lines = []

    all_sys_names = [
        "ap-abstractive-oracle.json",
        "agglo-abstractive-oracle.json",
        "baseline-oracle.json",
        "baseline-oracle-submod.json",
        "extractive-oracle.json"
    ]

    significant_diff_systems_tl17 = defaultdict(list)
    significant_diff_systems_crisis = defaultdict(list)

    if not use_tok:
        all_names = [n + "+sent" for n in all_sys_names]
    else:
        all_names = [n + "+tok" for n in all_sys_names]

    for sys_1, sys_2, symbol in [
        ("baseline-oracle.json", "baseline-oracle-submod.json", "\\circ"),
        ("baseline-oracle-submod.json", "agglo-abstractive-oracle.json", "*"),
        ("ap-abstractive-oracle.json", "agglo-abstractive-oracle.json", "\\dagger"),
        ("ap-abstractive-oracle.json", "extractive-oracle.json", "\\ddagger")
    ]:
        if use_tok:
            sys_1 += "+tok"
            sys_2 += "+tok"
        else:
            sys_1 += "+sent"
            sys_2 += "+sent"

        for entry, vals in create_sig_diff_dict(all_tl17_results, tl17_entries, sys_1, sys_2, symbol=symbol).items():
            significant_diff_systems_tl17[entry].extend(vals)
        for entry, vals in create_sig_diff_dict(all_crisis_results, crisis_entries, sys_1, sys_2, symbol=symbol).items():
            significant_diff_systems_crisis[entry].extend(vals)

    if use_tok:
        all_names = [n + "+tok" for n in all_sys_names]

        for system in all_sys_names:
            sig_level_tl17 = check_significance(all_tl17_results, system + "+tok", system + "+sent")
            sig_level_crisis = check_significance(all_crisis_results, system + "+tok", system + "+sent")

            for metric in metrics.split():
                if sig_level_tl17[metric] < 0.05:
                    symbol = "+"
                    if getattr(tl17_entries[system + "+tok"], metric).f1 < getattr(tl17_entries[system + "+sent"], metric).f1:
                        symbol = "-"
                    significant_diff_systems_tl17.setdefault((system + "+tok", metric), []).append(symbol)

                symbol = "^" + symbol
                if sig_level_crisis[metric] < 0.05:
                    symbol = "+"
                    if getattr(crisis_entries[system + "+tok"], metric).f1 < getattr(crisis_entries[system + "+sent"], metric).f1:
                        symbol = "-"
                    significant_diff_systems_crisis.setdefault((system + "+tok", metric), []).append(symbol)


    lines.append("\\multicolumn{{{}}}{{|l|}}{{\\textbf{{{}}}}}\\\\\\hline".format(len(metrics.split()) + 1, "Timeline 17"))
    #lines.extend(gen_table_part_for_corpus(all_names, tl17_entries))
    lines.extend(gen_latex_oracle_part(tl17_entries, all_names, significant_diff_systems_tl17))
    lines.append("\\hline\multicolumn{{{}}}{{|l|}}{{\\textbf{{{}}}}}\\\\\\hline".format(len(metrics.split())  + 1, "Crisis"))
    lines.extend(gen_latex_oracle_part(crisis_entries, all_names, significant_diff_systems_crisis))



    print("\n".join(lines))


def gen_latex_oracle_part(entries, all_names, significant_diff_systems):
    lines = []
    all_cells = np.empty(shape=(len(all_names), len(metrics.split())))
    for idx, system in enumerate(all_names):
        entry = entries[system]
        all_cells[idx,:] = entry.datesel.f1, entry.rouge_1_concat.f1, entry.rouge_2_concat.f1, entry.rouge_1_agree.f1, entry.rouge_2_agree.f1, entry.rouge_1_align.f1, entry.rouge_2_align.f1

    col_maxima = np.max(all_cells, axis=0)

    for system, row in zip(all_names, all_cells):
        name = ""
        if system.startswith("ap"):
            name = "Affinity Propagation"
        elif system.startswith("agglo"):
            name = "Agglomerative"
        elif system.startswith("extractive"):
            name = "Extractive"

        data_cells = []
        for col_idx, cell_val in enumerate(row):
            mark_bold = False
            if cell_val == col_maxima[col_idx]:
                mark_bold = True

            cell_text = None

            if mark_bold:
                cell_text = "\\textbf{{{:.3f}}}".format(cell_val)
            else:
                cell_text = "{:.3f}".format(cell_val)

            if (system, metrics.split()[col_idx]) in significant_diff_systems:
                cell_text += "$^{{{}}}$".format("".join(significant_diff_systems[(system, metrics.split()[col_idx])]))

            data_cells.append(cell_text)

        lines.append("{} & {}\\\\\\hline".format(
            name,
            "&".join(data_cells)
        ))

    return lines


def gen_latex_table_sent(tl17_entries, crisis_entries, all_tl17_results, all_crisis_results):
    systems = [
      #  "ap-abstractive-datetr-dateref-path.json+sent",
     #   "ap-abstractive-temptr-dateref-clsize-path.json+sent",
     #   "ap-abstractive-datetr-dateref.json+sent",
     #   "ap-abstractive-temptr-dateref-clsize.json+sent",
        "ap-abstractive-temptr.json+sent",
        "ap-abstractive-datetr-noclsize.json+sent",
        "ap-abstractive-noclsize.json+sent",
        "ap-abstractive-globaltr.json+sent",
        "agglo-abstractive-temptr.json+sent",
        "agglo-abstractive-datetr-noclsize.json+sent",
        "agglo-abstractive-noclsize.json+sent",
        "agglo-abstractive-globaltr.json+sent",
        "baseline-submod.json+sent",
        "baseline.json+sent"
    ]

    all_names = sorted(filter(lambda n: n in systems, set(tl17_entries.keys()) & set(crisis_entries.keys())))

    print(all_names)

    lines = []

    tl17_sig_diff_sys = dict()
    crisis_sig_diff_sys = dict()

    for cl_algo in "ap", "agglo":
        for config in [
            "abstractive-temptr.json+sent",
            "abstractive-datetr-noclsize.json+sent",
            "abstractive-noclsize.json+sent",
            "abstractive-globaltr.json+sent"
        ]:
            sys_1 = "ap-" + config
            sys_2 = "agglo-" + config
            #if check_significance(all_tl17_results, sys_1, sys_2) < 0.05:
            #    tl17_sig_pairs[(sys_1, sys_2)] = "*"
            #if check_significance(all_crisis_results, sys_1, sys_2) < 0.05:
            #    crisis_sig_pairs[(sys_1, sys_2)] = "*"

            tl17_sig_diff_sys.update(create_sig_diff_dict(all_tl17_results, tl17_entries, sys_1, sys_2))
            crisis_sig_diff_sys.update(create_sig_diff_dict(all_crisis_results, crisis_entries, sys_1, sys_2))

        if cl_algo == "ap":
            doc_cl_symbol = "\\dagger"
        else:
            doc_cl_symbol = "\\ddagger"

        for entry, vals in create_sig_diff_dict(all_tl17_results, tl17_entries, cl_algo + "-abstractive-noclsize.json+sent", "baseline-submod.json+sent", symbol=doc_cl_symbol).items():
            tl17_sig_diff_sys.setdefault(entry, []).extend(vals)

        for entry, vals in create_sig_diff_dict(all_crisis_results, crisis_entries, cl_algo + "-abstractive-noclsize.json+sent", "baseline-submod.json+sent", symbol=doc_cl_symbol).items():
            crisis_sig_diff_sys.setdefault(entry, []).extend(vals)

        for entry, vals in create_sig_diff_dict(all_tl17_results, tl17_entries, cl_algo + "-abstractive-temptr.json+sent", cl_algo + "-abstractive-datetr-noclsize.json+sent", symbol="a").items():
            tl17_sig_diff_sys.setdefault(entry, []).extend(vals)

        for entry, vals in create_sig_diff_dict(all_crisis_results, crisis_entries, cl_algo + "-abstractive-temptr.json+sent", cl_algo + "-abstractive-datetr-noclsize.json+sent", symbol="a").items():
            crisis_sig_diff_sys.setdefault(entry, []).extend(vals)

        for entry, vals in create_sig_diff_dict(all_tl17_results, tl17_entries, cl_algo + "-abstractive-temptr.json+sent", cl_algo + "-abstractive-globaltr.json+sent", symbol="b").items():
            tl17_sig_diff_sys.setdefault(entry, []).extend(vals)

        for entry, vals in create_sig_diff_dict(all_crisis_results, crisis_entries, cl_algo + "-abstractive-temptr.json+sent", cl_algo + "-abstractive-globaltr.json+sent", symbol="b").items():
            crisis_sig_diff_sys.setdefault(entry, []).extend(vals)

        for entry, vals in create_sig_diff_dict(all_tl17_results, tl17_entries, cl_algo + "-abstractive-temptr.json+sent", cl_algo + "-abstractive-noclsize.json+sent", symbol="c").items():
            tl17_sig_diff_sys.setdefault(entry, []).extend(vals)

        for entry, vals in create_sig_diff_dict(all_crisis_results, crisis_entries, cl_algo + "-abstractive-temptr.json+sent", cl_algo + "-abstractive-noclsize.json+sent", symbol="c").items():
            crisis_sig_diff_sys.setdefault(entry, []).extend(vals)


    for entry, vals in create_sig_diff_dict(all_tl17_results, tl17_entries, "baseline-submod.json+sent", "baseline.json+sent", symbol="\\circ").items():
        tl17_sig_diff_sys.setdefault(entry, []).extend(vals)

    for entry, vals in create_sig_diff_dict(all_crisis_results, crisis_entries, "baseline-submod.json+sent", "baseline.json+sent", symbol="\\circ").items():
        crisis_sig_diff_sys.setdefault(entry, []).extend(vals)




    longest_name_len = max(map(len, all_names))

    val_headers = "Date F1", "R1 concat", "R2 concat", "R1 agree", "R2 agree", "R1 align", "R2 align"

    lines.append(" & Date & \\multicolumn{ 2 }{|c|}{Concat} & \\multicolumn{ 2 }{|c|}{Agree} & \\multicolumn{ 2 }{|c|}{Align} \\\\")
    lines.append(" & F1 & R1 & R2 & R1 & R2 & R1 & R2 \\\\\hline\hline")

    #lines.append("\\hline {}\t& {}\\\\\\hline".format("System".ljust(longest_name_len), "\t& ".join(val_headers)))

    lines.append("\\multicolumn{{{}}}{{|l|}}{{ \\textbf{{{}}}}}\\\\\\hline".format(len(metrics.split()) + 1, "Timeline 17"))
    lines.extend(gen_table_part_for_corpus(all_names, tl17_entries, tl17_sig_diff_sys))
    lines.append("\\hline\multicolumn{{{}}}{{|l|}}{{ \\textbf{{{}}}}}\\\\\\hline".format(len(metrics.split())  + 1, "Crisis"))
    lines.extend(gen_table_part_for_corpus(all_names, crisis_entries, crisis_sig_diff_sys))

    print("\n".join(lines))


def create_sig_diff_dict(results, entries, sys_1, sys_2, symbol="*"):
    result = {}
    #print(sys_1, sys_2)
    sig_level = check_significance(results, sys_1, sys_2)

    #if symbol in "abc":
    #    print(sys_1, sys_2, sig_level)

    #print(sys_1, sys_2, sig_level)

    print(sig_level)
    for metric in metrics.split():
        if sig_level[metric] < 0.05:
            if getattr(entries[sys_2], metric).f1 > getattr(entries[sys_1], metric).f1:
                system = sys_2
            else:
                system = sys_1

            result[system, metric] = [symbol]
    return result


def gen_latex_table_sent_features(tl17_entries, crisis_entries, all_tl17_results, all_crisis_results):
    systems = [
      #  "ap-abstractive-datetr-dateref-path.json+sent",
     #   "ap-abstractive-temptr-dateref-clsize-path.json+sent",
     #   "ap-abstractive-datetr-dateref.json+sent",
     #   "ap-abstractive-temptr-dateref-clsize.json+sent",
        "ap-abstractive-temptr-dateref-clsize.json+sent",
        "ap-abstractive-temptr-dateref-clsize-path.json+sent",
        "ap-abstractive-datetr-dateref.json+sent",
        "ap-abstractive-datetr-dateref-path.json+sent",
        "ap-abstractive-globaltr-dateref-clsize.json+sent",
        "ap-abstractive-globaltr-dateref-clsize-path.json+sent",
        "chieu.json+sent"
    ]

    tl17_sig_diff_sys = {}
    crisis_sig_diff_sys = {}

    for base_system in [
        "ap-abstractive-temptr-dateref-clsize",
        "ap-abstractive-datetr-dateref",
        "ap-abstractive-globaltr-dateref-clsize"
    ]:
        sys_1 = base_system + ".json+sent"
        sys_2 = base_system + "-path.json+sent"

        tl17_sig_diff_sys.update(create_sig_diff_dict(all_tl17_results, tl17_entries, sys_1, sys_2, symbol="*"))
        crisis_sig_diff_sys.update(create_sig_diff_dict(all_crisis_results, crisis_entries, sys_1, sys_2, symbol="*"))

    for system in  ["ap-abstractive-temptr-dateref-clsize.json+sent",
            "ap-abstractive-temptr-dateref-clsize-path.json+sent",
            "ap-abstractive-datetr-dateref.json+sent",
            "ap-abstractive-datetr-dateref-path.json+sent",
            "ap-abstractive-globaltr-dateref-clsize.json+sent",
            "ap-abstractive-globaltr-dateref-clsize-path.json+sent"]:
        tl17_sig_diff_sys.update(create_sig_diff_dict(all_tl17_results, tl17_entries, system, "chieu.json+sent", symbol="\\circ"))

    for sys_1, sys_2 in [
        ["ap-abstractive-temptr-dateref-clsize.json+sent", "ap-abstractive-temptr.json+sent"],
        ["ap-abstractive-datetr-dateref.json+sent", "ap-abstractive-datetr-noclsize.json+sent"],
        ["ap-abstractive-globaltr-dateref-clsize.json+sent", "ap-abstractive-globaltr.json+sent"],
    ]:
        for entry, vals in create_sig_diff_dict(all_tl17_results, tl17_entries, sys_1, sys_2, symbol="\\dagger").items():
            tl17_sig_diff_sys.setdefault(entry, []).extend(vals)

        for entry, vals in create_sig_diff_dict(all_crisis_results, crisis_entries, sys_1, sys_2, symbol="\\dagger").items():
            crisis_sig_diff_sys.setdefault(entry, []).extend(vals)

    for sys_1, sys_2, symbol in [
        ["ap-abstractive-temptr-dateref-clsize-path.json+sent", "ap-abstractive-datetr-dateref-path.json+sent", "a"],
        ["ap-abstractive-datetr-dateref-path.json+sent", "ap-abstractive-globaltr-dateref-clsize-path.json+sent", "b"],
        #["ap-abstractive-globaltr-dateref-clsize.json+sent", "ap-abstractive-globaltr-noclsize.json+sent"],
    ]:
        for entry, vals in create_sig_diff_dict(all_tl17_results, tl17_entries, sys_1, sys_2, symbol=symbol).items():
            tl17_sig_diff_sys.setdefault(entry, []).extend(vals)

        for entry, vals in create_sig_diff_dict(all_crisis_results, crisis_entries, sys_1, sys_2, symbol=symbol).items():
            crisis_sig_diff_sys.setdefault(entry, []).extend(vals)


    all_names = sorted(filter(lambda n: n in systems, set(tl17_entries.keys()) & set(crisis_entries.keys())))

    lines = []

    lines.append(" & Date & \\multicolumn{ 2 }{|c|}{Concat} & \\multicolumn{ 2 }{|c|}{Agree} & \\multicolumn{ 2 }{|c|}{Align} \\\\")
    lines.append(" & F1 & R1 & R2 & R1 & R2 & R1 & R2 \\\\\hline\hline")

    #lines.append("\\hline {}\t& {}\\\\\\hline".format("System".ljust(longest_name_len), "\t& ".join(val_headers)))

    lines.append("\\multicolumn{{{}}}{{|l|}}{{ \\textbf{{{}}}}}\\\\\\hline".format(len(metrics.split()) + 1, "Timeline 17"))
    lines.extend(gen_table_part_for_corpus(all_names, tl17_entries, tl17_sig_diff_sys))
    lines.append("\\hline\multicolumn{{{}}}{{|l|}}{{ \\textbf{{{}}}}}\\\\\\hline".format(len(metrics.split()) + 1, "Crisis"))
    lines.extend(gen_table_part_for_corpus(all_names, crisis_entries, crisis_sig_diff_sys))

    print("\n".join(lines))


def gen_latex_table_simple(systems, comparison_pairs, tl17_entries, crisis_entries, all_tl17_results, all_crisis_results):
    #systems = [
    #  #  "ap-abstractive-datetr-dateref-path.json+sent",
    # #   "ap-abstractive-temptr-dateref-clsize-path.json+sent",
    # #   "ap-abstractive-datetr-dateref.json+sent",
    # #   "ap-abstractive-temptr-dateref-clsize.json+sent",
    #    "ap-abstractive-temptr-dateref-clsize.json+sent",
    #    "ap-abstractive-temptr-dateref-clsize-path.json+sent",
    #    "ap-abstractive-datetr-dateref.json+sent",
    #    "ap-abstractive-datetr-dateref-path.json+sent",
    #    "ap-abstractive-globaltr-dateref-clsize.json+sent",
    #    "ap-abstractive-globaltr-dateref-clsize-path.json+sent",
    #    "chieu.json+sent"
    #]

    tl17_sig_diff_sys = {}
    crisis_sig_diff_sys = {}

    for sys_1, sys_2, symbol in comparison_pairs:
        for key, entry in create_sig_diff_dict(all_tl17_results, tl17_entries, sys_1, sys_2, symbol=symbol).items():
            tl17_sig_diff_sys.setdefault(key, []).extend(entry)
        #print("WARNING! crisis sig diff disabled!")
        for key, entry in create_sig_diff_dict(all_crisis_results, crisis_entries, sys_1, sys_2, symbol=symbol).items():
            crisis_sig_diff_sys.setdefault(key, []).extend(entry)

    lines = []

    lines.append(" & Date & \\multicolumn{ 2 }{|c|}{Concat} & \\multicolumn{ 2 }{|c|}{Agree} & \\multicolumn{ 2 }{|c|}{Align} \\\\")
    lines.append(" & F1 & R1 & R2 & R1 & R2 & R1 & R2 \\\\\hline\hline")

    #lines.append("\\hline {}\t& {}\\\\\\hline".format("System".ljust(longest_name_len), "\t& ".join(val_headers)))

    lines.append("\\multicolumn{{{}}}{{|l|}}{{ \\textbf{{{}}}}}\\\\\\hline".format(len(metrics.split()) + 1, "Timeline 17"))
    lines.extend(gen_table_part_for_corpus(systems, tl17_entries, tl17_sig_diff_sys))
    lines.append("\\hline\multicolumn{{{}}}{{|l|}}{{ \\textbf{{{}}}}}\\\\\\hline".format(len(metrics.split()) + 1, "Crisis"))
    lines.extend(gen_table_part_for_corpus(systems, crisis_entries, crisis_sig_diff_sys))

    print("\n".join(lines))



def gen_table_part_for_corpus(all_names, all_result_entries, significant_diff_systems=set(), names=None):
    lines = []
    all_cells = np.empty(shape=(len(all_names), len(metrics.split())))
    for idx, (system, name) in enumerate(all_names):
        entry = all_result_entries[system]
        all_cells[idx,:] = entry.datesel.f1, entry.rouge_1_concat.f1, entry.rouge_2_concat.f1, entry.rouge_1_agree.f1, entry.rouge_2_agree.f1, entry.rouge_1_align.f1, entry.rouge_2_align.f1

    all_cells = np.round(all_cells, 3)

    col_maxima = np.max(all_cells, axis=0)

    previous_clusterer_name = ""

    for idx, ((system, name), row) in enumerate(zip(all_names, all_cells)):
        data_cells = []

        for col_idx, cell_val in enumerate(row):
            mark_bold = False
            if cell_val == col_maxima[col_idx]:
                mark_bold = True

            cell_str = "{:.3f}".format(cell_val)

            if mark_bold:
                cell_str = "\\textbf{{{}}}".format(cell_str)

            if (system, metrics.split()[col_idx]) in significant_diff_systems:
                cell_str += "$^{{{}}}$".format("".join(significant_diff_systems[system, metrics.split()[col_idx]]))

            data_cells.append(cell_str)

        if names is None and False:
            if "chieu" not in system:
                cl_method, score_func = clusterer_and_score_func_name_from_system_description(system)

                if previous_clusterer_name != cl_method:
                    previous_clusterer_name = cl_method
                    lines.append("\multicolumn{{{}}}{{|l|}}{{ \\textit{{{}}} }}\\\\\\hline".format(len(metrics.split()) + 1, previous_clusterer_name))
            else:
                score_func = "Chieu"

            lines.append("{} \t & {} \\\\\\hline".format(score_func, "\t& ".join(data_cells)))
        else:
            lines.append("{} \t & {} \\\\\\hline".format(name, "\t& ".join(data_cells)))

    return lines


def gen_table_part_for_corpus_tok(all_names, all_result_entries, significant_diff_systems, names=None):
    lines = []
    all_cells_sent = np.empty(shape=(len(all_names), len(metrics.split())))
    all_cells_tok = np.empty(shape=(len(all_names), len(metrics.split())))
    for idx, system in enumerate(all_names):
        entry_sent = all_result_entries[system + "+sent"]
        entry_tok = all_result_entries[system + "+tok"]
        all_cells_sent[idx,:] = entry_sent.datesel.f1, entry_sent.rouge_1_concat.f1, entry_sent.rouge_2_concat.f1, entry_sent.rouge_1_agree.f1, entry_sent.rouge_2_agree.f1, entry_sent.rouge_1_align.f1, entry_sent.rouge_2_align.f1
        all_cells_tok[idx,:] = entry_tok.datesel.f1, entry_tok.rouge_1_concat.f1, entry_tok.rouge_2_concat.f1, entry_tok.rouge_1_agree.f1, entry_tok.rouge_2_agree.f1, entry_tok.rouge_1_align.f1, entry_tok.rouge_2_align.f1

    all_cells_sent = np.round(all_cells_sent, 3)
    all_cells_tok = np.round(all_cells_tok, 3)

    col_maxima = np.max(all_cells_tok, axis=0)


    #\multirow{ 2}{*}{1}

    previous_clusterer_name = ""

    for idx, (system, sent_row, tok_row) in enumerate(zip(all_names, all_cells_sent, all_cells_tok)):
        cl_method, score_func = clusterer_and_score_func_name_from_system_description(system)

        if previous_clusterer_name != cl_method:
            previous_clusterer_name = cl_method
            lines.append("\multicolumn{{{}}}{{|l|}}{{ \\textit{{{}}} }}\\\\\\hline".format(len(metrics.split()) + 1, previous_clusterer_name))

        data_cells = ["\\multirow{{2}}{{*}}{{{}}}".format(score_func)]
        change_cells = [""]

        for col_idx, cell_val in enumerate(tok_row):
            mark_bold = False
            if cell_val == col_maxima[col_idx]:
                mark_bold = True

            if mark_bold:
                data_cells.append("\\textbf{{{:.3f}}}".format(cell_val))
            else:
                data_cells.append("{:.3f}".format(cell_val))

            diff = tok_row[col_idx] - sent_row[col_idx]

            change_cell = None

            if diff == 0:
                change_cell = "{:.3f}".format(diff)
            elif diff < 0:
                change_cell = ("\\small \\textcolor{{red}}{{{:+#.3f}}}".format(diff))
            else:
                change_cell = ("\\small \\textcolor{{green}}{{{:+#.3f}}}".format(diff))

            if (system, metrics.split()[col_idx]) in significant_diff_systems:
                change_cell = "\\underline{{{}}}".format(change_cell)

            change_cells.append(change_cell)

        lines.append("{} \\\\".format("\t& ".join(data_cells)))
        lines.append("{} \\\\\\hline".format("\t& ".join(change_cells)))

    return lines



def gen_latex_table_tok(tl17_entries, crisis_entries, all_tl17_results, all_crisis_results):
    #all_names = sorted(set(map(lambda x: x.split("+")[0], filter(lambda n: "oracle" not in n, set(tl17_entries.keys()) & set(crisis_entries.keys())))))

    all_names = [
        "ap-abstractive-datetr-dateref.json",
        "ap-abstractive-datetr-dateref-path.json",
        "ap-abstractive-globaltr-dateref-clsize.json",
        "ap-abstractive-globaltr-dateref-clsize-path.json",
        "ap-abstractive-temptr-dateref-clsize.json",
        "ap-abstractive-temptr-dateref-clsize-path.json",
    ]


    significant_diff_systems_tl17 = set()
    significant_diff_systems_crisis = set()
    for system in all_names:
        sig_level_tl17 = check_significance(all_tl17_results, system + "+tok", system + "+sent")
        sig_level_crisis = check_significance(all_crisis_results, system + "+tok", system + "+sent")

        for metric in metrics.split():
            if sig_level_tl17[metric] < 0.025:
                significant_diff_systems_tl17.add((system, metric))
            if sig_level_crisis[metric] < 0.025:
                significant_diff_systems_crisis.add((system, metric))

    lines = []

    lines.append(" & Date & \\multicolumn{ 2 }{|c|}{Concat} & \\multicolumn{ 2 }{|c|}{Agree} & \\multicolumn{ 2 }{|c|}{Align} \\\\")
    lines.append(" & F1 & R1 & R2 & R1 & R2 & R1 & R2 \\\\\hline\hline")

    #lines.append("\\hline {}\t& {}\\\\\\hline".format("System".ljust(longest_name_len), "\t& ".join(val_headers)))

    lines.append("\multicolumn{{{}}}{{|l|}}{{ \\textbf{{{}}} }}\\\\\\hline".format(len(metrics.split()) + 1, "Timeline 17"))
    lines.extend(gen_table_part_for_corpus_tok(all_names, tl17_entries, significant_diff_systems_tl17))
    lines.append("\multicolumn{{{}}}{{|l|}}{{ \\textbf{{{}}} }}\\\\\\hline".format(len(metrics.split()) + 1, "Crisis"))
    lines.extend(gen_table_part_for_corpus_tok(all_names, crisis_entries, significant_diff_systems_crisis))

    print("\n".join(lines))


def scorer_names_from_parts(parts):
    return parts
    scorer_names = ["ling"]
    for elem in parts[1:]:
        if elem == "abstractive" or elem == "datesel":
            continue

        if elem == "temptr":
            scorer_names.append("TR^{(temp)}")
        elif elem == "globaltr":
            scorer_names.append("TR^{(global)}")
        elif elem == "dateref":
            scorer_names.append("\\text{dateref}")
        #elif elem == "clsize":
        #    scorer_names.append("\\text{clsize}")
        elif elem == "path":
            scorer_names.append("\\text{path}")
        elif elem == "clsize":
            scorer_names.append("\\text{csize}")
        elif elem == "datetr":
            scorer_names.append("TR^{(date)}")
        elif elem == "noclsize":
            continue
        else:
            raise ValueError("Unknown component {}".format(elem))
    return scorer_names


def clusterer_and_score_func_name_from_system_description(sys_key):
    system_config_name = sys_key.split("+")[0].split(".")[0]

    parts = system_config_name.split("-")

    clustering_method = None

    clustering_method_names = {"agglo": "Agglomerative Clustering", "ap": "Affinity Propagation"}

    if parts[0] == "baseline":
        if len(parts) == 2 and parts[1] == "submod":
            scorer_names = ["\\text{ling}", "TR^{(cluster)}", "submod"]
        else:
            scorer_names = ["\\text{ling}", "TR^{(cluster)}"]
        clustering_method = "Document-Guided Clustering"
    elif len(parts) == 2 and parts[1] == "abstractive":
        scorer_names = ["\\text{ling}", "TR^{(cluster)}", "\\text{cluster size}"]
        clustering_method = clustering_method_names[parts[0]]
    elif len(parts) == 3 and parts[1] == "abstractive" and parts[2] == "noclsize":
        scorer_names = ["\\text{ling}", "TR^{(cluster)}"]
        clustering_method = clustering_method_names[parts[0]]
    elif len(parts) == 3 and parts[1] == "abstractive" and parts[2] == "datesel":
        scorer_names = ["\\text{ling}", "TR^{(cluster)}", "\\text{cluster size}"]
        clustering_method = clustering_method_names[parts[0]] + "Date Selection"
        scorer_names = scorer_names_from_parts(parts)
    else:
        clustering_method = clustering_method_names[parts[0]]
        scorer_names = scorer_names_from_parts(parts)

    return clustering_method, "$f_{{{}}}$".format("*".join(scorer_names))

#
#    for sys_name, entry in sorted(all_entries.items(), key=lambda i: i[1].rouge_2_concat.f1, reverse=True):
#        if "+sent" not in sys_name:
#            continue
#
#        cells = [sys_name.ljust(longest_name_len)]
#
#        for val, header in zip((entry.datesel.f1, entry.rouge_1_concat.f1, entry.rouge_2_concat.f1, entry.rouge_1_agree.f1, entry.rouge_2_agree.f1, entry.rouge_1_align.f1, entry.rouge_2_align.f1), val_headers):
#            cells.append("{:.3f}".format(val).ljust(len(header)))
#
#        print("\t".join(cells))


if __name__ == "__main__":
    analyze_main()
