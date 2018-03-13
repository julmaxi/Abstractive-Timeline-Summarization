from tilse.data.timelines import Timeline
from tilse.evaluation import rouge

import os

from utils import iter_files, iter_dirs, fst, scnd
from graphsum import SentenceCompressionGraph

from collections import namedtuple, Counter, defaultdict
import sys
import datetime

import logging

from graphsum import summarize_timeline_dir, read_clusters, save_cluster

import pickle

from langmodel import KenLMLanguageModel
from reader import DatedSentenceReader, DatedTimelineCorpusReader
from clustering import generate_affinity_matrix_from_dated_sentences, write_similarity_file, read_ap_file, cluster_sentences_ap
from similarity import SklearnTfIdfCosineSimilarityModel
from submodular import RedundancyFactor, CoverageFactor, KnapsackConstraint, SubsetKnapsackConstraint, SubModularOptimizer, MaxDateCountConstraint, ClusterMembershipConstraint

import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from graphsum import ClusterGenerator, STOPWORDS, generate_summary_candidates, calculate_keyword_text_rank
from similarity import BinaryOverlapSimilarityModel

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

TimelineParameters = namedtuple("TimelineParameters", "first_date last_date max_date_count max_sent_count max_date_sent_count")

import nltk
import collections

class RougeReimplementation:
    def __init__(self, stem=True, ignore_stopwords=True):
        self.stem = stem
        self.ignore_stopwords = ignore_stopwords
        self.stopwords = set()
        self.porter_stemmer = nltk.stem.PorterStemmer()

        self.stem_function = self.identity

        if stem:
            self.stem_function = self.robust_porter_stemmer

        dir_path = os.path.dirname(os.path.realpath(__file__))

        if ignore_stopwords:
            with open("./venv/lib/python3.5/site-packages/pyrouge/tools/ROUGE-1.5.5/data/smart_common_words.txt") as my_file:
                self.stopwords = set(my_file.read().splitlines())

    def score_summary(self, summary, references):
        punctuation = [".", ",", ";", ":", "``", "''", "-", '"']

        to_ignore = self.stopwords.union(punctuation)

        pred_tokens_lowercased = [self.stem_function(k.lower()) for sent in summary for k in sent
                                  if k.lower() not in to_ignore]

        ref_tokens_lowercased = {}

        for i, ref_summary in references.items():
            ref_tokens_lowercased[i] = [self.stem_function(k.lower()) for sent in ref_summary for k
                                        in sent if k.lower() not in to_ignore]

        eval_scores = {}
        eval_scores.update(
            self._rouge_1(pred_tokens_lowercased, ref_tokens_lowercased))
        eval_scores.update(
            self._rouge_2(pred_tokens_lowercased, ref_tokens_lowercased))

        return eval_scores

    def identity(self, x):
        return x

    def robust_porter_stemmer(self, x):
        stem = x

        try:
            stem = self.porter_stemmer.stem(x)
        except IndexError:
            pass

        return stem

    def _rouge_1(self, pred_tokens, ref_tokens):
        # unigrams
        pred_counts = collections.Counter(pred_tokens)

        ref_counts = {}

        for i, tokens in ref_tokens.items():
            ref_counts[i] = collections.Counter(tokens)

        # approximate ROUGE-1 score
        match = 0
        for tok in pred_counts:
            match += sum([min(pred_counts[tok], ref_counts[x][tok]) for x in
                          ref_counts.keys()])

        prec_denom = (len(ref_counts.keys()) * sum(pred_counts.values()))

        recall_denom = sum([sum(ref_counts[x].values()) for x in ref_counts])

        return {
            "rouge_1_h_count": match,
            "rouge_1_p_count": prec_denom,
            "rouge_1_m_count": recall_denom,
        }

    def _rouge_2(self, pred_tokens, ref_tokens):
        pred_counts = collections.Counter(zip(pred_tokens, pred_tokens[1:]))

        ref_counts = {}

        for i, tokens in ref_tokens.items():
            ref_counts[i] = collections.Counter(zip(tokens, tokens[1:]))

        # approximate ROUGE-1 score
        match = 0
        for tok in pred_counts:
            match += sum([min(pred_counts[tok], ref_counts[x][tok]) for x in
                          ref_counts.keys()])

        prec_denom = (len(ref_counts.keys()) * sum(pred_counts.values()))

        recall_denom = sum([sum(ref_counts[x].values()) for x in ref_counts])

        return {
            "rouge_2_h_count": match,
            "rouge_2_p_count": prec_denom,
            "rouge_2_m_count": recall_denom,
        }


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
        constraints.append(SubsetKnapsackConstraint(parameters.max_date_sent_count, dict((i, 1) for i in range(sent_idx_counter)), member_ids))

    constraints.append(MaxDateCountConstraint(parameters.max_date_count, sent_id_date_map))
    constraints.append(KnapsackConstraint(parameters.max_sent_count, dict((i, 1) for i in range(sent_idx_counter))))

    if disallow_cluster_repetition:
        constraints.append(ClusterMembershipConstraint(id_cluster_map))

    #cluster_redundancy_factor = RedundancyFactor(id_score_map, id_cluster_map)

    print("Selecting from {} sentences".format(sent_idx_counter))
    print("Initializing redudancy")
    factors = []
    kmeans_redundancy_factor = RedundancyFactor.from_sentences(
        id_score_map,
        id_sentence_map,
        num_clusters=min(max(sent_idx_counter // 5, 2), 2 * parameters.max_date_sent_count * parameters.max_date_count))
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


class SentenceScorer:
    def __init__(self, config):
        self.lm = KenLMLanguageModel.from_file("langmodel20k_vp_3.bin")

        method = config.get("method")

        self.use_lm = False
        self.use_local_informativeness = False
        self.use_cluster_size = False

        if method is None:
            self.use_lm = config.get("use_lm", True)
            self.use_local_informativeness = config.get("use_local_info", True)
            self.use_cluster_size = config.get("use_cluster_size", False)

        elif method == "cluster_size":
            self.use_lm = False
            self.use_local_informativeness = False
            self.use_cluster_size = True

    def prepare(self, corpus):
        pass

    def prepare_for_clusters(self, clusters):
        if not self.use_cluster_size:
            return

        self.max_cluster_size = max(map(lambda c: len(c), clusters))

    def score_function_for_cluster(self, cluster):
        local_tr_scores = calculate_keyword_text_rank([s.as_token_tuple_sequence("form", "pos") for s in cluster])

        def calc_local_informativeness(sent):
            local_informativeness_score = 0
            for token in set(sent):
                local_informativeness_score += local_tr_scores.get(token, 0)

            return local_informativeness_score

        def sfunc(sent, info):
            score = 1

            if self.use_local_informativeness:
                local_informativeness_score = calc_local_informativeness(sent)
                score *= local_informativeness_score

            if self.use_lm:
                lm_score = 1 / (1.0 - self.lm.estimate_sent_log_proba(" ".join(map(lambda t: t[0], sent))))
                score *= lm_score

            if self.use_cluster_size:
                score *= len(cluster) / self.max_cluster_size

            return score

        return sfunc



class TLSumModuleBase:
    def __init__(self, config):
        pass

    def prepare(self, corpus):
        pass


class GraphCandidateGenerator(TLSumModuleBase):
    def __init__(self, config):
        self.use_weighting = config.get("use_weighting", False)
        self.maxlen = config.get("sent_maxlen", 55)

    def prepare(self, corpus):
        self.tfidf_model = TfidfVectorizer()
        self.tfidf_model.fit(list(map(lambda s: s.as_tokenized_string(), corpus.sentences)))

    def generate_candidates(self, cluster):
        compressor = SentenceCompressionGraph(STOPWORDS)
        compressor.add_sentences([sent.as_token_tuple_sequence("form_lowercase", "pos") for sent in cluster])

        cluster_vectors = self.tfidf_model.transform(list(map(lambda s: s.as_tokenized_string(), cluster)))

        def check_closeness(sent):
            sent_vec = self.tfidf_model.transform([" ".join(map(lambda x: x[0], sent))])
            sims = cosine_similarity(sent_vec, cluster_vectors)

            return all(sims[0,:] <= 0.8)

        for proposed_sent, path_weight in compressor.generate_compression_candidates(
                filterfunc=check_closeness,
                use_weighting=self.use_weighting,
                maxlen=self.maxlen,
                return_weight=True):

            yield proposed_sent, {"weight": path_weight}


class CentroidCandidateGenerator(TLSumModuleBase):
    def __init__(self, config):
        pass

    def prepare(self, corpus):
        self.tfidf = TfidfVectorizer()
        self.tfidf.fit(list(map(lambda s: s.as_tokenized_string(), corpus.sentences)))

    def generate_candidates(self, cluster):
        vecs = self.tfidf.transform(list(map(lambda s: s.as_tokenized_string(), cluster)))
        sims = cosine_similarity(vecs)

        best_idx = sims.sum(0).argmax()

        return [(cluster[best_idx].as_token_tuple_sequence("form", "pos"), {})]


class MayorityClusterDater:
    def __init__(self, config):
        pass

    def date_cluster(self, cluster):
        #print(list(map(lambda s: s.predicted_date, cluster)))
        #return cluster[0].predicted_date
        referenced_dates = Counter()

        for sent in cluster:
            if len(sent.exact_date_references) > 0:
                referenced_dates.update(sent.exact_date_references)
            elif len(sent.all_date_tags) == 0:
                referenced_dates.update([sent.document.dct_tag])

        print(referenced_dates)

        if len(referenced_dates) == 0:
            return None
        cluster_tag, _ = referenced_dates.most_common(1)[0]
        cluster_date = datetime.date(cluster_tag.year, cluster_tag.month, cluster_tag.day)

        return cluster_date


class APClusterer:
    def __init__(self, config):
        self.include_uncertain_date_edges = config.get("include_uncertain_date_edges", True)
        self.predicted_tag_only = config.get("predicted_tag_only", False)

    def cluster_corpus(self, corpus):
        return cluster_sentences_ap(
            corpus.sentences,
            include_uncertain_date_edges=self.include_uncertain_date_edges,
            predicted_tag_only=self.predicted_tag_only)


class IdentityClusterer:
    def __init__(self, config):
        pass

    def cluster_corpus(self, corpus):
        return [[sent] for sent in corpus.sentences]


class AgglomerativeClusterer:
    def __init__(self, config):
        self.clustering_threshold = config.get("clustering_threshold", 0.3)

    def cluster_corpus(self, corpus):
        date_sentences = defaultdict(list)
        logger.debug("Partitioning {} sentences".format(len(corpus.sentences)))
        for sentence in corpus.sentences:
            date_sentences[sentence.predicted_date].append(sentence)

        logger.debug("Fitting Tfidf Model on {} documents".format(corpus.num_documents))
        sim_model = TfidfVectorizer()
        sim_model.fit(list(map(lambda d: d.plaintext, corpus)))

        all_clusters = []

        for date, sentences in date_sentences.items():
            logger.debug("Clustering {} sentences for date {}".format(len(sentences), date))
            all_clusters.extend(self.cluster_date_sentences(sentences, sim_model))

        return all_clusters

    def cluster_date_sentences(self, sentences, sim_model):
        sent_vecs = sim_model.transform(list(map(lambda s: s.as_tokenized_string(), sentences)))

        similarities = cosine_similarity(sent_vecs)

        clusters = []

        for row in similarities:
            cluster = []
            _, indices, _ = scipy.sparse.find(row > self.clustering_threshold)
            for idx in indices:
                cluster.append(sentences[idx])

            if len(cluster) > 1:
                clusters.append(set(cluster))

        non_redundant_clusters = []

        import networkx as nx
        from networkx.algorithms.components import connected_components

        graph = nx.Graph()

        for idx, cluster_1 in enumerate(clusters):
            for idx2, cluster_2 in enumerate(clusters[idx + 1:]):
                overlap = len(cluster_1 & cluster_2)
                if overlap / len(cluster_1) >= 1.0 or overlap / len(cluster_2) >= 1.0:
                    graph.add_edge(idx, idx2)

        for component in connected_components(graph):
            new_cluster = set()

            for subcluster_idx in component:
                new_cluster.update(clusters[subcluster_idx])

            non_redundant_clusters.append(list(new_cluster))

        #for idx, cluster_1 in enumerate(clusters):
        #    if cluster_1 is None:
        #        continue
        #    to_merge = []
        #    for idx_2, cluster_2 in enumerate(clusters[idx + 1:]):
        #        if cluster_2 is None:
        #            continue
        #        if len(cluster_1 & cluster_2) / len(cluster_1 | cluster_2) > 0.8:
        #            to_merge.append((idx_2, cluster_2))
#
        #    for idx_2, cluster_2 in to_merge:
        #        cluster_1.update(cluster_2)
        #        clusters[idx_2] = None
        #        non_redundant_clusters.append(list(cluster_1))

        return non_redundant_clusters


        #for idx, sent_vec in enumerate(sent_vecs):
        #    for seed_vec, cluster in clusters:
        #        if seed_vec.dot(sent_vec.T)[0] > self.clustering_threshold:
        #            cluster.append(sentences[idx])
        #    clusters.append((sent_vec, [sentences[idx]]))

        return list(map(lambda x: x[1], clusters))




class GlobalSubModularSentenceSelector:
    def __init__(self, config):
        pass

    def prepare(self, corpus):
        self.corpus = corpus

    def select_sentences_from_clusters(self, per_date_clusters, parameters):
        return select_tl_sentences_submod(per_date_clusters, self.corpus, parameters)


class IdentityCandidateGenerator:
    def __init__(self, config):
        pass

    def prepare(self, corpus):
        pass

    def generate_candidates(self, cluster):
        return [(s.as_token_tuple_sequence("form", "pos"), None) for s in cluster]


class ROUGEOracleScorer(TLSumModuleBase):
    def __init__(self, config):
        pass

    def prepare_for_clusters(self, clusters):
        pass

    def score_clusters_for_timeline(self, per_date_cluster_candidates, timeline):
        scored_per_date_cluster_candidates = {}
        rouge_eval = RougeReimplementation()

        for date, clusters in per_date_cluster_candidates:
            reference = {"A": [s.split() for s in  timeline[date]]}
            scored_clusters = []
            for cluster in clusters:
                scored_sentences = []
                for sent, info in cluster:
                    if date in timeline.get_dates():
                        scores = rouge_eval.score_summary([list(map(fst, sent))], reference)
                        if scores["rouge_2_p_count"] > 0:
                            prec = scores["rouge_2_h_count"] / scores["rouge_2_p_count"]
                        else:
                            prec = 0.0
                        if scores["rouge_2_m_count"] > 0:
                            rec = scores["rouge_2_h_count"] / scores["rouge_2_m_count"]
                        else:
                            rec = 0.0
                        if prec + rec > 0:
                            score = (prec * rec) / (prec + rec)
                        else:
                            score = 0.0
                    else:
                        score = 0.0

                    scored_sentences.append((sent, score))
                scored_clusters.append(scored_sentences)
            scored_per_date_cluster_candidates[date] = scored_clusters

        return list(scored_per_date_cluster_candidates.items())


class GloballyClusteredSentenceCompressionTimelineGenerator:
    def clusterer_from_config(self, config):
        method = config["method"]

        if method == "ap":
            return APClusterer(config)
        elif method == "agglo":
            return AgglomerativeClusterer(config)
        elif method == "identity":
            return IdentityClusterer(config)
        else:
            raise ValueError("Method {!r} not recognized".format(method))

    def selector_from_config(self, config):
        method = config["method"]

        if method == "global_submod":
            return GlobalSubModularSentenceSelector(config)
        else:
            raise ValueError("Method {!r} not recognized".format(method))

    def generator_from_config(self, config):
        method = config.get("method", "graph")

        if method == "graph":
            return GraphCandidateGenerator(config)
        elif method == "centroid":
            return CentroidCandidateGenerator(config)
        elif method == "identity":
            return IdentityCandidateGenerator(config)
        else:
            raise ValueError("Method {!r} not recognized".format(method))

    def scorer_from_config(self, config):
        method = config.get("method", "default")
        if method == "default":
            return SentenceScorer(config.get("scoring", {}))
        elif method == "rouge_oracle":
            return ROUGEOracleScorer(config)
        else:
            raise ValueError("Method {!r} not recognized".format(method))


    def __init__(self, config):
        self.config = config

        self.clusterer = self.clusterer_from_config(config["clustering"])
        self.scorer = self.scorer_from_config(config.get("scoring", {}))
        self.generator = self.generator_from_config(config.get("generation", {}))
        self.cluster_dater = MayorityClusterDater(config.get("dating", {}))
        self.sentence_selector = self.selector_from_config(config["selection"])

        self.min_cluster_size = config.get("min_cluster_size", 0)

    def generate_timelines(self, corpus, all_parameters, reference_timelines=None):
        self.scorer.prepare(corpus)
        self.generator.prepare(corpus)
        self.sentence_selector.prepare(corpus)

        clusters = self.create_clusters(corpus)

        per_date_cluster_candidates = defaultdict(list)

        self.scorer.prepare_for_clusters(clusters)

        logger.debug("Processing {} clusters".format(len(clusters)))

        for cluster in clusters:
            if len(cluster) < self.min_cluster_size:
                continue

            #for item in cluster:
            #    print(item.as_tokenized_string())
            #print("-----")

            logger.debug("Processing cluster with size {}".format(len(cluster)))

            cluster_candidates = []

            cluster_date = self.cluster_dater.date_cluster(cluster)
            if cluster_date is None:
                continue

            candidates_and_info = self.generator.generate_candidates(cluster)
            if hasattr(self.scorer, "score_function_for_cluster"):
                score_func = self.scorer.score_function_for_cluster(cluster)
                for candidate, info in candidates_and_info:
                    score = score_func(candidate, info)
                    cluster_candidates.append((candidate, score))
            else:
                cluster_candidates = candidates_and_info

            per_date_cluster_candidates[cluster_date].append(cluster_candidates)

        per_date_cluster_candidates = list(per_date_cluster_candidates.items())

        all_timelines = []
        for timeline_idx, parameters in enumerate(all_parameters):
            if hasattr(self.scorer, "score_clusters_for_timeline") and reference_timelines is not None:
                # For oracle summarization
                per_date_cluster_candidates = self.scorer.score_clusters_for_timeline(per_date_cluster_candidates, reference_timelines[timeline_idx])

            timeline = self.sentence_selector.select_sentences_from_clusters(per_date_cluster_candidates, parameters)
            all_timelines.append(Timeline(timeline))

        return all_timelines

    def create_clusters(self, corpus):
        base_cache_path = self.config.get("cluster_cache_path")
        if base_cache_path is None:
            clusters = self.clusterer.cluster_corpus(corpus)
            return clusters
        cache_path = os.path.join(base_cache_path, corpus.name)

        if not os.path.isdir(cache_path):
            clusters = self.clusterer.cluster_corpus(corpus)
            os.makedirs(cache_path)

            for idx, cluster in enumerate(clusters):
                save_cluster(os.path.join(cache_path, "{}.txt".format(idx)), cluster)
        else:
            clusters = list(read_clusters(cache_path, corpus).values())

        return clusters




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
