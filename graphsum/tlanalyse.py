from argparse import ArgumentParser

from utils import iter_files, iter_dirs

import os

from collections import namedtuple, defaultdict

import art.scores
import art.significance_tests
import art.aggregators

metrics = "datesel rouge_1_concat rouge_2_concat rouge_1_agree rouge_2_agree rouge_1_align rouge_2_align"

ResultEntry = namedtuple("ResultEntry", metrics)
FMeasureEntry = namedtuple("FMeasureEntry", "recall precision f1")


def print_results_table(all_entries):
    all_names = all_entries.keys()

    longest_name_len = max(map(len, all_names))

    val_headers = "Date F1", "R1 concat", "R2 concat", "R1 agree", "R2 agree", "R1 align", "R2 align"

    print("{}\t{}".format("System".ljust(longest_name_len), "\t".join(val_headers)))

    for sys_name, entry in sorted(all_entries.items(), key=lambda i: i[1].rouge_2_concat.f1, reverse=True):
        if "+sent" not in sys_name:
            continue

        cells = [sys_name.ljust(longest_name_len)]

        for val, header in zip((entry.datesel.f1, entry.rouge_1_concat.f1, entry.rouge_2_concat.f1, entry.rouge_1_agree.f1, entry.rouge_2_agree.f1, entry.rouge_1_align.f1, entry.rouge_2_align.f1), val_headers):
            cells.append("{:.3f}".format(val).ljust(len(header)))

        print("\t".join(cells))


def check_significance(all_entries, key_1, key_2):
    all_timeline_names = sorted(set([(topic_key, tl_key) for topic_key, tls in all_entries[key_1].items() for tl_key in tls.keys()]))

    all_scores = defaultdict(lambda: [[], []])
    for topic_name, tl_name in all_timeline_names:
        for score in metrics.split():
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
    results_basedir = "evaluation_results"

    all_entries = {}
    tl17_entries = {}
    crisis_entries = {}

    all_tl17_results = {}
    all_crisis_results = {}

    for system_dir in iter_dirs(results_basedir):
        entry, tl17_entry, crisis_entry, tl17_results, crisis_results = analyze_system_results_dir(system_dir)
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

    print_results_table(all_entries)
    print("\n===== TL17 =====")
    print_results_table(tl17_entries)
    print("\n===== Crisis =====")
    print_results_table(crisis_entries)

    #gen_latex_table_sent(tl17_entries, crisis_entries)
    #print()
    #gen_latex_table_tok(tl17_entries, crisis_entries, all_tl17_results, all_crisis_results)

    significance_pairs = [
        ("ap-abstractive-temptr-dateref-clsize-path.json", "ap-abstractive-datetr-dateref-path.json")
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
        
                all_results = []
                for metric in metrics.split():
                    result = results[metric]
                    if result <= 0.05:
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


def analyze_system_results_dir(results_dir):
    all_results = {}

    tl17_results = {}
    crisis_results = {}

    relevant_files = list(iter_files(results_dir, ".txt"))

    if len(relevant_files) == 0:
        return None, None, None, None, None

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

    return macro_average_entry, tl17_macro_average_entry, crisis_macro_average_entry, tl17_results, crisis_results


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

def gen_latex_table_sent(tl17_entries, crisis_entries):
    all_names = sorted(filter(lambda n: "+sent" in n and "oracle" not in n, set(tl17_entries.keys()) & set(crisis_entries.keys())))

    lines = []

    longest_name_len = max(map(len, all_names))

    val_headers = "Date F1", "R1 concat", "R2 concat", "R1 agree", "R2 agree", "R1 align", "R2 align"

    lines.append(" & Date & \\multicolumn{ 2 }{|c|}{Concat} & \\multicolumn{ 2 }{|c|}{Agree} & \\multicolumn{ 2 }{|c|}{Align} \\\\")
    lines.append(" & F1 & R1 & R2 & R1 & R2 & R1 & R2 \\\\\hline\hline")

    #lines.append("\\hline {}\t& {}\\\\\\hline".format("System".ljust(longest_name_len), "\t& ".join(val_headers)))

    lines.append("\\multicolumn{{ {} }}{{|l|}}{{ \\textbf{{ {} }} }}\\\\\\hline".format(len(metrics.split()) + 1, "Timeline 17"))
    lines.extend(gen_table_part_for_corpus(all_names, tl17_entries))
    lines.append("\\hline\multicolumn{{ {} }}{{|l|}}{{ \\textbf{{ {} }} }}\\\\\\hline".format(len(metrics.split())  + 1, "Crisis"))
    lines.extend(gen_table_part_for_corpus(all_names, crisis_entries))

    print("\n".join(lines))


def gen_table_part_for_corpus(all_names, all_result_entries):
    lines = []
    all_cells = np.empty(shape=(len(all_names), len(metrics.split())))
    for idx, system in enumerate(all_names):
        entry = all_result_entries[system]
        all_cells[idx,:] = entry.datesel.f1, entry.rouge_1_concat.f1, entry.rouge_2_concat.f1, entry.rouge_1_agree.f1, entry.rouge_2_agree.f1, entry.rouge_1_align.f1, entry.rouge_2_align.f1

    col_maxima = np.max(all_cells, axis=0)

    previous_clusterer_name = ""

    for system, row in zip(all_names, all_cells):
        data_cells = []

        for col_idx, cell_val in enumerate(row):
            mark_bold = False
            if cell_val == col_maxima[col_idx]:
                mark_bold = True

            if mark_bold:
                data_cells.append("\\textbf{{ {:.3f} }}".format(cell_val))
            else:
                data_cells.append("{:.3f}".format(cell_val))

        cl_method, score_func = clusterer_and_score_func_name_from_system_description(system)

        if previous_clusterer_name != cl_method:
            previous_clusterer_name = cl_method
            lines.append("\multicolumn{{ {} }}{{|l|}}{{ \\textit{{ {} }} }}\\\\\\hline".format(len(metrics.split()) + 1, previous_clusterer_name))

        lines.append("{} \t & {} \\\\\\hline".format(score_func, "\t& ".join(data_cells)))

    return lines


def gen_table_part_for_corpus_tok(all_names, all_result_entries, significant_diff_systems):
    lines = []
    all_cells_sent = np.empty(shape=(len(all_names), len(metrics.split())))
    all_cells_tok = np.empty(shape=(len(all_names), len(metrics.split())))
    for idx, system in enumerate(all_names):
        entry_sent = all_result_entries[system + "+sent"]
        entry_tok = all_result_entries[system + "+tok"]
        all_cells_sent[idx,:] = entry_sent.datesel.f1, entry_sent.rouge_1_concat.f1, entry_sent.rouge_2_concat.f1, entry_sent.rouge_1_agree.f1, entry_sent.rouge_2_agree.f1, entry_sent.rouge_1_align.f1, entry_sent.rouge_2_align.f1
        all_cells_tok[idx,:] = entry_tok.datesel.f1, entry_tok.rouge_1_concat.f1, entry_tok.rouge_2_concat.f1, entry_tok.rouge_1_agree.f1, entry_tok.rouge_2_agree.f1, entry_tok.rouge_1_align.f1, entry_tok.rouge_2_align.f1

    col_maxima = np.max(all_cells_tok, axis=0)


    #\multirow{ 2}{*}{1}

    previous_clusterer_name = ""

    for system, sent_row, tok_row in zip(all_names, all_cells_sent, all_cells_tok):
        cl_method, score_func = clusterer_and_score_func_name_from_system_description(system)

        if previous_clusterer_name != cl_method:
            previous_clusterer_name = cl_method
            lines.append("\multicolumn{{ {} }}{{|l|}}{{ \\textit{{ {} }} }}\\\\\\hline".format(len(metrics.split()) + 1, previous_clusterer_name))

        data_cells = ["\\multirow{{2}}{{*}}{{{}}}".format(score_func)]
        change_cells = [""]

        for col_idx, cell_val in enumerate(tok_row):
            mark_bold = False
            if cell_val == col_maxima[col_idx]:
                mark_bold = True

            if mark_bold:
                data_cells.append("\\textbf{{ {:.3f} }}".format(cell_val))
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
    all_names = sorted(set(map(lambda x: x.split("+")[0], filter(lambda n: "oracle" not in n, set(tl17_entries.keys()) & set(crisis_entries.keys())))))

    significant_diff_systems_tl17 = set()
    significant_diff_systems_crisis = set()
    for system in all_names:
        sig_level_tl17 = check_significance(all_tl17_results, system + "+tok", system + "+sent")
        sig_level_crisis = check_significance(all_crisis_results, system + "+tok", system + "+sent")

        for metric in metrics.split():
            if sig_level_tl17[metric] < 0.05:
                significant_diff_systems_tl17.add((system, metric))
            if sig_level_crisis[metric] < 0.05:
                significant_diff_systems_crisis.add((system, metric))

    lines = []

    lines.append(" & Date & \\multicolumn{ 2 }{|c|}{Concat} & \\multicolumn{ 2 }{|c|}{Agree} & \\multicolumn{ 2 }{|c|}{Align} \\\\")
    lines.append(" & F1 & R1 & R2 & R1 & R2 & R1 & R2 \\\\\hline\hline")

    #lines.append("\\hline {}\t& {}\\\\\\hline".format("System".ljust(longest_name_len), "\t& ".join(val_headers)))

    lines.append("\multicolumn{{ {} }}{{|l|}}{{ \\textbf{{ {} }} }}\\\\\\hline".format(len(metrics.split()) + 1, "Timeline 17"))
    lines.extend(gen_table_part_for_corpus_tok(all_names, tl17_entries, significant_diff_systems_tl17))
    lines.append("\multicolumn{{ {} }}{{|l|}}{{ \\textbf{{ {} }} }}\\\\\\hline".format(len(metrics.split()) + 1, "Crisis"))
    lines.extend(gen_table_part_for_corpus_tok(all_names, crisis_entries, significant_diff_systems_crisis))

    print("\n".join(lines))


def scorer_names_from_parts(parts):
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
        else:
            raise ValueError("Unknown component {}".format(elem))
    return scorer_names


def clusterer_and_score_func_name_from_system_description(sys_key):
    system_config_name = sys_key.split("+")[0].split(".")[0]

    parts = system_config_name.split("-")

    clustering_method = None

    clustering_method_names = {"agglo": "Agglomerative Clustering", "ap": "Affinity Propagation"}

    if parts[0] == "baseline":
        scorer_names = ["\\text{ling}", "TR^{(cluster)}"]
        clustering_method = "Document-Guided Clustering"
    elif len(parts) == 2 and parts[1] == "abstractive":
        scorer_names = ["\\text{ling}", "TR^{(cluster)}", "\\text{cluster size}"]
        clustering_method = clustering_method_names[parts[0]]
    elif len(parts) == 3 and parts[1] == "abstractive" and parts[2] == "datesel":
        scorer_names = ["\\text{ling}", "TR^{(cluster)}", "\\text{cluster size}"]
        clustering_method = clustering_method_names[parts[0]] + "Date Selection"
        scorer_names = scorer_names_from_parts(parts)
    else:
        clustering_method = clustering_method_names[parts[0]]
        scorer_names = scorer_names_from_parts(parts)

    return clustering_method, "$f_{{ {} }}$".format("*".join(scorer_names))

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
