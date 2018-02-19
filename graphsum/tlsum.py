from tilse.data import timelines
from tilse.evaluation import rouge

import os

from utils import iter_files, iter_dirs, fst, scnd

from collections import namedtuple, Counter, defaultdict
import sys
import datetime

from graphsum import summarize_timeline_dir

from langmodel import KenLMLanguageModel
from reader import DatedSentenceReader, DatedTimelineCorpusReader

TimelineParameters = namedtuple("TimelineParameters", "first_date last_date max_date_count max_sent_count max_date_sent_count")


def determine_parameters(gold_dir):
    global_earliest_date = None
    global_last_date = None

    max_tl_date_count = 0

    max_tl_date_sent_count = 0
    max_tl_sent_count = 0

    all_timelines = []
    for fpath in iter_files(gold_dir, ".txt"):
        with open(fpath, encoding="latin-1") as f:
            timeline = timelines.Timeline.from_file(f)
        all_timelines.append(timeline)

        dateset = timeline.get_dates()
        earliest_date = min(*dateset)
        if global_earliest_date is None or earliest_date < global_earliest_date:
            global_earliest_date = earliest_date
        last_date = max(*dateset)
        if global_last_date is None or global_last_date > global_last_date:
            global_last_date = last_date

        max_tl_date_count = max(len(dateset), max_tl_date_count)

        total_sent_len = 0
        for date in timeline:
            sents = timeline[date]
            total_sent_len += len(sents)

            max_tl_date_sent_count = max(max_tl_date_sent_count, len(sents))

        max_tl_sent_count = max(total_sent_len, max_tl_sent_count)

    return TimelineParameters(
        global_earliest_date,
        global_last_date,
        max_tl_date_count,
        max_tl_sent_count,
        max_tl_date_sent_count)

import pickle

def run_full_tl_summ(timeline_func):
    params = determine_parameters(sys.argv[2])
    with open(sys.argv[1], "rb") as f:
        corpus = pickle.load(f)
    tl = timeline_func(corpus, params)
    with open("timeline.txt", "w") as f_out:
        f_out.write(str(tl))


def select_best_date_by_doc_freq(document_dir, parameters):
    doc_counter = Counter()
    for dir_ in iter_dirs(document_dir):
        date = date_from_dirname(os.path.basename(dir_))
        if date is None:
            continue
        if date <= parameters.last_date and date >= parameters.first_date:
            print(dir_)
            doc_counter[date] = len(list(iter_files(dir_, "cont")))

    dates = list(map(fst, doc_counter.most_common(parameters.max_date_count)))
    return dates


def create_timeline(document_dir, timeml_dir, parameters):
    dates = select_best_date_by_doc_freq(document_dir, parameters)

    date_summary_dict = {}

    lm = KenLMLanguageModel.from_file("langmodel20k_vp_3.bin")
    for date in dates:
        date_path = os.path.join(document_dir, date.strftime("%Y-%m-%d"))
        summarization = summarize_timeline_dir(date_path, lm)

        if len(summarization.strip()) > 0:
            date_summary_dict[date] = summarization.split("\n")

    return timelines.Timeline(date_summary_dict)


def date_from_dirname(dirname):
    parts = dirname.split("-")
    assert len(parts) == 3, "Dirname must have format YYYY-mm-dd"
    year, month, day = parts
    if int(day) == 0:
        return None

    return datetime.date(int(year), int(month), int(day))

from submodular import RedundancyFactor, CoverageFactor, KnapsackConstraint, SubsetKnapsackConstraint, SubModularOptimizer

def select_tl_sentences_submod(per_date_cluster_candidates, doc_sents, max_sents_per_date=2):
    id_sentence_map = {}
    id_cluster_map = {}
    date_id_map = defaultdict(list)
    id_date_map = {}
    id_score_map = {}
    id_tok_count_map = {}
    sent_id_date_map = {}

    sent_idx_counter = 0
    cluster_idx_counter = 0

    for date_idx, (date, clusters) in enumerate(per_date_cluster_candidates):
        for members in clusters:
            for sent, score in members:
                id_sentence_map[sent_idx_counter] = sent
                id_cluster_map[sent_idx_counter] = cluster_idx_counter
                date_id_map[date_idx].append(sent_idx_counter)
                id_date_map[sent_idx_counter] = date_idx
                id_score_map[sent_idx_counter] = score
                id_tok_count_map[sent_idx_counter] = len(sent)
                sent_id_date_map[sent_idx_counter] = date

                sent_idx_counter += 1

            cluster_idx_counter += 1

    constraints = []

    for date_id, member_ids in date_id_map.items():
        constraints.append(SubsetKnapsackConstraint(max_sents_per_date, defaultdict(lambda: 1), member_ids))

    #cluster_redundancy_factor = RedundancyFactor(id_score_map, id_cluster_map)

    print("Selecting from {} sentences".format(sent_idx_counter))
    print("Initializing redudancy")
    kmeans_redundancy_factor = RedundancyFactor.from_sentences(
        id_score_map,
        id_sentence_map,
        num_clusters=max_sents_per_date * len(per_date_cluster_candidates) * 5)

    #print("Initializing coverage")
    #coverage_factor = CoverageFactor.from_sentences(
    #    doc_sents,
    #    list(map(scnd, sorted(id_sentence_map.items(), key=fst)))
    #)

    opt = SubModularOptimizer(
        [
            kmeans_redundancy_factor,
 #           coverage_factor
        ],
        constraints)

    print("Running Optimizier")
    sent_ids = opt.run(range(sent_idx_counter))

    selected_tl_sentences = defaultdict(list)
    for sent_id in sent_ids:
        date = sent_id_date_map[sent_id]
        sent = " ".join([tok for tok, pos in id_sentence_map[sent_id]])
        selected_tl_sentences[date].append(sent)

    return selected_tl_sentences



from graphsum import ClusterGenerator, STOPWORDS, generate_summary_candidates, calculate_keyword_text_rank
from similarity import BinaryOverlapSimilarityModel


def cluster_without_seed_sent_level(sentences):
    sim_model = BinaryOverlapSimilarityModel(STOPWORDS)
    clusters = []

    for sent in sentences:
        for seed, members in clusters:
            cos_sim = sim_model.compute_similarity(
                seed.as_token_attr_sequence("form"),
                sent.as_token_attr_sequence("form")
            )

            if cos_sim > 0.7:
                members.append(sent)

        clusters.append((sent, [sent]))

    return dict(enumerate(map(scnd, clusters)))


def eliminate_duplicate_clusters(clusters):
    base_clusters = []
    for cluster in clusters:
        cluster = set(cluster)
        matched_any = False
        for base_cluster in base_clusters:
            if (len(base_cluster.intersection(cluster))) / len((base_cluster.union(cluster))) == 1.0:
                base_cluster = base_cluster.union(cluster)
                matched_any = True

        if not matched_any:
            base_clusters.append(cluster)

    return dict(enumerate(base_clusters))


def create_timeline_sentence_level(timeline_corpus, parameters):
    reader = DatedSentenceReader()

    date_ref_counts = Counter()

    #timeline_corpus = DatedTimelineCorpusReader().run(document_dir, timeml_dir)

    sents_by_date = defaultdict(list)

    for doc in timeline_corpus:
        for sent in doc:
            date_ref_counts.update(sent.exact_date_references)
            sents_by_date[sent.predicted_date].append(sent)

    lm = KenLMLanguageModel.from_file("langmodel20k_vp_3.bin")
    global_tr = calculate_keyword_text_rank([sent.as_token_tuple_sequence("form", "pos") for sents in timeline_corpus for sent in sents])

    best_dates = list(map(lambda t: t[0], date_ref_counts.most_common(parameters.max_date_count)))

    cluster_gen = ClusterGenerator(cluster_without_seed_sent_level, 5)

    per_date_candidates = []

    for date in best_dates:
        date_candidates = []
        sents = sents_by_date.get(date, [])
        clusters = list(cluster_gen.cluster_from_documents(timeline_corpus, cache_key=date.strftime("%Y-%m-%d"), clustering_input=sents).values())

        clusters = eliminate_duplicate_clusters(clusters).values()

        if len(clusters) == 0:
            continue

        clusters = sorted(clusters, key=lambda c: len(c))[:10]

        for cluster_sents in clusters:
            candidates = generate_summary_candidates(
                    list(
                        map(lambda s: s.as_token_tuple_sequence("form_lowercase", "pos"),
                            cluster_sents)), lm,
                    length_normalized=True,
                    tr_scores=global_tr)
            date_candidates.append(candidates)

        per_date_candidates.append((date, date_candidates))

    date_summary_dict = select_tl_sentences_submod(
        per_date_candidates,
        [sent.as_token_tuple_sequence("form_lowercase", "pos") for doc in documents for sent in doc]
    )

    return timelines.Timeline(date_summary_dict)


from clustering import generate_affinity_matrix_from_dated_sentences, write_similarity_file
from similarity import SklearnTfIdfCosineSimilarityModel


def create_timeline_clustering(document_dir, timeml_dir, parameters):
    reader = DatedSentenceReader()

    all_sents = []
    all_doc_texts = []

    for date_dir in sorted(iter_dirs(document_dir)):
        dir_date = datetime.datetime.strptime(os.path.basename(date_dir), "%Y-%m-%d").date()

        for doc_fname in iter_files(date_dir, ".tokenized"):
            timeml_fname = os.path.join(timeml_dir, os.path.basename(date_dir), os.path.basename(doc_fname) + ".timeml")
            doc = reader.read(doc_fname, timeml_fname, dir_date)

            all_doc_texts.append(doc.as_token_attr_sequence("form"))

            for sent in doc.sentences:
                all_sents.append(sent)

    sim_model = SklearnTfIdfCosineSimilarityModel(stem=False)
    sim_model.fit(all_doc_texts)

    affinities = generate_affinity_matrix_from_dated_sentences(all_sents, sim_model)
    write_similarity_file("similarities.txt", affinities)


if __name__ == "__main__":
    run_full_tl_summ(create_timeline_sentence_level)
