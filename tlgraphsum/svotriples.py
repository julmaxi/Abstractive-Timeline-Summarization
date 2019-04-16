import tlgraphsum.reader as reader
import sys
import json
from tlgraphsum.tlsum import GloballyClusteredSentenceCompressionTimelineGenerator
from tilse.data.timelines import Timeline
from collections import defaultdict, Counter
from tlgraphsum.params import determine_tl_parameters

def main():
    base_config_path = sys.argv[1]
    corpus_fname = sys.argv[2]
    gold_tl_fname = sys.argv[3]

    with open(gold_tl_fname, errors="ignore") as f:
        gold_tl = Timeline.from_file(f)

    with open(base_config_path) as f:
        config = json.load(f)

    corpus = reader.load_corpus(corpus_fname)

    generator = GloballyClusteredSentenceCompressionTimelineGenerator(config)

    corpus_promise, cluster_promise, dated_cluster_promise, cluster_candidates = generator.get_promises(corpus)

    dated_clusters = dated_cluster_promise.get()

    all_svo_tuples_per_date = defaultdict(Counter)

    for cluster, date in dated_clusters:
        for sentence in cluster:
            for pred, subj, obj in sentence.dependency_tree.extract_svo_tuples():
                if pred is not None:
                    pred = pred.lemma.lower()
                if obj is not None:
                    obj = obj.lemma.lower()
                if subj is not None:
                    subj = subj.lemma.lower()

                all_svo_tuples_per_date[date][(pred, subj, obj)] += 1

    triples_by_frequency = []

    for date, counter in sorted(all_svo_tuples_per_date.items()):
        for triple, count in counter.items():
            triples_by_frequency.append((count, date, triple))

    params = determine_tl_parameters(gold_tl)

    selected_triples = dict()

    for (cnt, date, triple) in sorted(triples_by_frequency, reverse=True, key=lambda x: x[0]):
        if date < params.first_date or date > params.last_date:
            continue

        if date not in selected_triples:
            if len(selected_triples) < params.max_date_count:
                selected_triples[date] = []
            else:
                continue

        if len(selected_triples[date]) > params.max_date_sent_count:
            continue

        if triple[0] in ("happen", "say"):
            continue

        selected_triples[date].append(str(triple))

    print(Timeline(selected_triples))

if __name__ == "__main__":
    main()
