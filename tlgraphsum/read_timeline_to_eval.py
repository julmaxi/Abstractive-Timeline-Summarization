import argparse
from tilse.data.timelines import Timeline
from collections import defaultdict

import os

from tlgraphsum.tleval import write_results_file
from tlgraphsum.params import determine_tl_parameters

def read_gold_tl_dir(gold_dir):
    timelines = {}
    for tl_fname in os.listdir(gold_dir):
        tl_path = os.path.join(gold_dir, tl_fname)

        with open(tl_path, errors="ignore") as f:
            timelines[tl_fname] = Timeline.from_file(f)

    return timelines


def parse_tl_name(tl_fname):
    tl_fname = os.path.basename(tl_fname)
    corpus, tl_name = tl_fname[:-len(".json.tl.txt")].split("__")

    return corpus, tl_name


def ensure_is_dir(dirpath):
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)


def timeline_by_applying_constraints(sys_tl, params, constraint_type="tok"):
    if constraint_type == "sent":
        for date, summary in list(sys_tl.dates_to_summaries.items()):
            if constraint_type == "tok":
                new_summary = apply_token_constraint_to_text("\n".join(summary), params.max_token_count)
            elif constraint_type == "sent":
                new_summary = apply_sent_constraint_to_text("\n".join(summary), params.max_sent_count)
            else:
                raise ValueError()

            sys_tl.dates_to_summaries[date] = new_summary
    return sys_tl


import spacy

nlp = spacy.load('en')


def apply_token_constraint_to_text(text, max_toks):
    doc = nlp(text)

    return [" ".join([t.text for t in doc][:max_toks])]


def apply_sent_constraint_to_text(text, max_sents):
    # Load English tokenizer, tagger, parser, NER and word vectors
    doc = nlp(text)

    sents = list(doc.sents)[:max_sents]

    return [" ".join([t.text for s in sents for t in s])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", dest="evaluation_results_dir", default="evaluation_results")
    parser.add_argument("-s", dest="system_timelines_dir", default="system_timelines")
    parser.add_argument("-g", dest="gold_timelines_dir", default="gold-timelines")

    parser.add_argument("-c", dest="cutoff_constraint", default="none")

    parser.add_argument("system_name")
    parser.add_argument("system_tl_files", nargs="+")

    args = parser.parse_args()

    per_corpus_timelines = defaultdict(dict)

    for tl_fname in args.system_tl_files:
        corpus_name, tl_name = parse_tl_name(tl_fname)

        with open(tl_fname) as f:
            timeline = Timeline.from_file(f)
            per_corpus_timelines[corpus_name][tl_name] = timeline

    system_name = args.system_name + "+" + args.cutoff_constraint

    print(system_name)

    system_sys_tl_dir = os.path.join(args.system_timelines_dir, system_name)

    system_evaluation_dir = os.path.join(args.evaluation_results_dir, system_name)

    ensure_is_dir(system_sys_tl_dir)
    ensure_is_dir(system_evaluation_dir)

    for corpus_name, tls in per_corpus_timelines.items():
        corpus_sys_tl_dir = os.path.join(system_sys_tl_dir, corpus_name)

        ensure_is_dir(corpus_sys_tl_dir)

        gold_tl_dir = os.path.join(args.gold_timelines_dir, corpus_name[:-len(".pkl")])

        gold_tls = read_gold_tl_dir(gold_tl_dir)

        sys_tls = [it[1] for it in sorted(tls.items())]
        gold_tls = [it for it in sorted(gold_tls.items())]

        if args.cutoff_constraint != "none":
            sys_tls = [timeline_by_applying_constraints(sys_tl, determine_tl_parameters(gold_tl), constraint_type=args.cutoff_constraint) for sys_tl, (_, gold_tl) in zip(sys_tls, gold_tls)]

        write_results_file(os.path.join(system_evaluation_dir, corpus_name[:-len(".pkl")] + ".txt"), corpus_sys_tl_dir, gold_tls, sys_tls)

if __name__ == "__main__":
    main()