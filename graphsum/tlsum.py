from tilse.data.timelines import Timeline
from tilse.evaluation import rouge

import os

from utils import iter_files, iter_dirs, fst, scnd

from collections import namedtuple, Counter, defaultdict
import sys
import datetime

from graphsum import summarize_timeline_dir

from langmodel import KenLMLanguageModel
from reader import DatedSentenceReader, DatedTimelineCorpusReader
from clustering import generate_affinity_matrix_from_dated_sentences, write_similarity_file, read_ap_file, cluster_sentences_ap
from similarity import SklearnTfIdfCosineSimilarityModel
from submodular import RedundancyFactor, CoverageFactor, KnapsackConstraint, SubsetKnapsackConstraint, SubModularOptimizer, MaxDateCountConstraint, ClusterMembershipConstraint


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
            timeline = Timeline.from_file(f)
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

    return Timeline(date_summary_dict)


def date_from_dirname(dirname):
    parts = dirname.split("-")
    assert len(parts) == 3, "Dirname must have format YYYY-mm-dd"
    year, month, day = parts
    if int(day) == 0:
        return None

    return datetime.date(int(year), int(month), int(day))

def select_tl_sentences_submod(per_date_cluster_candidates, doc_sents, parameters, disallow_cluster_repetition=True):
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
        if date < parameters.first_date or date > parameters.last_date:
            continue

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
        constraints.append(SubsetKnapsackConstraint(parameters.max_date_sent_count, defaultdict(lambda: 1), member_ids))

    constraints.append(MaxDateCountConstraint(parameters.max_date_count, sent_id_date_map))
    constraints.append(KnapsackConstraint(parameters.max_sent_count, defaultdict(lambda: 1)))

    if disallow_cluster_repetition:
        constraints.append(ClusterMembershipConstraint(id_cluster_map))

    #cluster_redundancy_factor = RedundancyFactor(id_score_map, id_cluster_map)

    print("Selecting from {} sentences".format(sent_idx_counter))
    print("Initializing redudancy")
    factors = []
    kmeans_redundancy_factor = RedundancyFactor.from_sentences(
        id_score_map,
        id_sentence_map,
        num_clusters=max(sent_idx_counter // 800, 2))
    factors.append(kmeans_redundancy_factor)



    #print("Initializing coverage")
    #coverage_factor = CoverageFactor.from_sentences(
    #    doc_sents,
    #    list(map(scnd, sorted(id_sentence_map.items(), key=fst)))
    #)

    opt = SubModularOptimizer(
        [
            kmeans_redundancy_factor,
    #        coverage_factor
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
        [sent.as_token_tuple_sequence("form_lowercase", "pos") for doc in timeline_corpus for sent in doc]
    )

    return Timeline(date_summary_dict)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SentenceScorer:
    def __init__(self, config):
        self.lm = KenLMLanguageModel.from_file("langmodel20k_vp_3.bin")

    def prepare(self, corpus):
        pass

    def score_function_for_cluster(self, cluster):
        local_tr_scores = calculate_keyword_text_rank(cluster)

        def calc_local_informativeness(sent):
            local_informativeness_score = 0
            for token in set(sent):
                local_informativeness_score += tr_scores.get(token, 0)

            return local_informativeness_score

        def sfunc(sent):
            lm_score = 1 / (1.0 - self.lm.estimate_sent_log_proba(" ".join(sent)))

            return lm_score * local_informativeness_score

        return sfunc


class APClusteringTimelineGenerator:
    def __init__(self, config):
        self.extractive = config.get("extractive", False)

    def _score_clusters(self, clusters):
        scored_clusters = []
        maxlen = max(map(lambda c: len(c), clusters))
        for cluster in clusters:
            scored_clusters.append((cluster, len(cluster) / maxlen))

        return scored_clusters


    def generate_timelines(self, corpus, all_parameters):
        lm = KenLMLanguageModel.from_file("langmodel20k_vp_3.bin")

        clustering = cluster_sentences_ap(corpus.sentences)

        #ap_matrix = generate_affinity_matrix_from_dated_sentences()

        #clustering = read_ap_file("clustering.txt", corpus.sorted_sentences)
        per_date_candidates = defaultdict(list)

        tfidf = TfidfVectorizer()
        tfidf.fit([doc.plaintext for doc in corpus])

        clustering = self._score_clusters(clustering)

        for cluster, cluster_score in clustering:
            if len(cluster) < 5:
                continue
            referenced_dates = Counter()
            for sent in cluster:
                if len(sent.exact_date_references) > 0:
                    referenced_dates.update(sent.exact_date_references)
                elif len(sent.all_date_tags) == 0:
                    referenced_dates.update([sent.document.dct_tag])
            if len(referenced_dates) == 0:
                continue

            cluster_tag, _ = referenced_dates.most_common(1)[0]
            cluster_date = datetime.date(cluster_tag.year, cluster_tag.month, cluster_tag.day)

            if self.extractive:
                similarities = tfidf.fit_transform([sent.as_tokenized_string() for sent in cluster])
                sims = cosine_similarity(similarities)
                best_idx = sims.sum(1).argmax()

                first_date = None
                last_date = None

                for sent in cluster:
                    date = datetime.datetime(sent.document.dct_tag.year, sent.document.dct_tag.month, sent.document.dct_tag.day)
                    if first_date is None:
                        first_date = date
                        last_date = date

                    if first_date > date:
                        first_date = date

                    if last_date < date:
                        last_date = date

                    print(sent.as_tokenized_string())

                #score = max((last_date - first_date).days, 1)
                score = len(cluster)

                candidates = [(cluster[best_idx].as_token_tuple_sequence("form_lowercase", "pos"), score)]
            else:
                candidates = generate_summary_candidates(
                        list(
                            map(lambda s: s.as_token_tuple_sequence("form_lowercase", "pos"),
                                cluster)), lm,
                        length_normalized=False,
                        use_weighting=False)

                candidates = [(cand, cluster_score * sent_score) for cand, sent_score in candidates]

            per_date_candidates[cluster_date].append(candidates)

        timelines = []
        for params in all_parameters:
            print(params)
            timeline = select_tl_sentences_submod(
                list(per_date_candidates.items()),
                [sent.as_token_tuple_sequence("form_lowercase", "pos") for doc in corpus for sent in doc],
                params
            )
            timelines.append(Timeline(timeline))

            print(Timeline(timeline))

        return timelines


def create_timeline_clustering(corpus, parameters):
    reader = DatedSentenceReader()
    lm = KenLMLanguageModel.from_file("langmodel20k_vp_3.bin")

    #affinities = generate_affinity_matrix_from_dated_sentences(all_sents, sim_model)
    #affinities = generate_affinity_matrix_from_dated_sentences(corpus.sorted_sentences)
    #write_similarity_file("similarities.txt", len(corpus.sentences), affinities)

    clustering = read_ap_file("clustering.txt", corpus.sorted_sentences)

    per_date_candidates = defaultdict(list)
    for cluster in clustering:
        if len(cluster) < 5:
            continue
        referenced_dates = Counter()

        for sent in cluster:
            if len(sent.exact_date_references) > 0:
                referenced_dates.update(sent.exact_date_references)
            elif len(sent.all_date_tags) == 0:
                referenced_dates.update([sent.document.dct_tag])

        cluster_date = None
        for cluster_tag, _ in sorted(referenced_dates.items(), key=lambda i: i[1], reverse=True):
            cluster_date = datetime.date(cluster_tag.year, cluster_tag.month, cluster_tag.day)
            if cluster_date > parameters.last_date or cluster_date < parameters.first_date:
                cluster_date = None

        if cluster_date is None:
            continue

        candidates = generate_summary_candidates(
                list(
                    map(lambda s: s.as_token_tuple_sequence("form_lowercase", "pos"),
                        cluster)), lm,
                length_normalized=False,
                use_weighting=False)

        per_date_candidates[cluster_date].append(candidates)

    date_summary_dict = select_tl_sentences_submod(
        list(per_date_candidates.items()),
        [sent.as_token_tuple_sequence("form_lowercase", "pos") for doc in corpus for sent in doc],
        parameters
    )

    return timelines.Timeline(date_summary_dict)


if __name__ == "__main__":
    run_full_tl_summ(create_timeline_clustering)
