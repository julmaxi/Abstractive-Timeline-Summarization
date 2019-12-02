from tilse.data.timelines import Timeline

import os

from utils import iter_files, iter_dirs, fst, scnd
from graphsum import SentenceCompressionGraph

from collections import namedtuple, Counter, defaultdict
import sys
import datetime

import logging

import json
from urllib.parse import quote, unquote
import itertools as it

import multiprocessing
from multiprocessing import get_context, cpu_count
from collections.abc import Sequence

from graphsum import summarize_timeline_dir, read_clusters, save_cluster

import pickle

from langmodel import KenLMLanguageModel
from clustering import cluster_sentences_ap
from submodular import RedundancyFactor, SubsetKnapsackConstraint, SubModularOptimizer, MaxDateCountConstraint, ClusterMembershipConstraint, ConstantSizeSubsetKnapsackConstraint, BucketedCoverageFactor, BucketedRedundancyFactor, SimilarityConstraint, ConstantScoreFactor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from graphsum import STOPWORDS, calculate_keyword_text_rank
from reader import DateTag
from similarity import BinaryOverlapSimilarityModel

import nltk
import collections
from component import ConstantPromise, CacheManager
from datesel import compute_relative_date_frequencies

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

TimelineParameters = namedtuple("TimelineParameters", "first_date last_date max_date_count max_sent_count max_token_count max_date_sent_count")

logger = multiprocessing.log_to_stderr()
logger.setLevel(multiprocessing.SUBDEBUG)


def compute_temp_tr(per_date_tr_scores, at_dates=None, sents=None, use_cl_size_reweighting=False):
    per_date_temporalized_tr_scores = {}
    logger.debug("Computing temp tr")
    for date in per_date_tr_scores.keys():
        if at_dates is not None and date not in at_dates:
            continue

        weighted_tr_score_sums = defaultdict(lambda: 0)
        factor_sum = 0

        for other_date, tr_scores in per_date_tr_scores.items():
            factor = 1.0 / (abs((other_date - date).days) + 1) * len(sents[other_date])

            for term, score in tr_scores.items():
                weighted_tr_score_sums[term] += score * factor

            factor_sum += score * factor

        per_date_temporalized_tr_scores[date] = dict((term, weight / factor_sum) for term, weight in weighted_tr_score_sums.items())

    return per_date_temporalized_tr_scores


class RougeReimplementation:
    def __init__(self, stem=True, ignore_stopwords=True):
        self.stem = stem
        self.ignore_stopwords = ignore_stopwords
        self.stopwords = set()
        self.porter_stemmer = nltk.stem.PorterStemmer()

        self.stem_function = self.identity

        if stem:
            self.stem_function = self.robust_porter_stemmer

        if ignore_stopwords:
            with open("./libs/pyrouge/tools/ROUGE-1.5.5/data/smart_common_words.txt") as my_file:
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


class Cluster(Sequence):
    def __init__(self, members):
        if isinstance(members, tuple) and len(members) == 2 and isinstance(members[1], DateTag):
            self.members = members[0]
            self.date = members[1]
        else:
            self.members = members
            self.date = None
        self.context = {}

    def __getitem__(self, idx):
        return self.members[idx]

    def __len__(self):
        return len(self.members)

    def as_token_tuple_sequence(self, *args):
        return [m.as_token_tuple_sequence(*args) for m in self.members]


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


def select_tl_sentences_greedy(per_date_cluster_candidates, doc_sents, parameters, disallow_cluster_repetition=False):
    id_sentence_map = {}
    id_score_map = {}
    id_cluster_map = {}
    sent_id_date_map = {}
    date_id_map = defaultdict(list)
    id_tok_count_map = {}

    sent_idx_counter = 0
    cluster_idx_counter = 0

    for date_idx, (date, clusters) in enumerate(per_date_cluster_candidates):
        if date < parameters.first_date or date > parameters.last_date:
            continue

        for members in clusters:
            for sent, score in members:
                id_cluster_map[sent_idx_counter] = cluster_idx_counter
                id_sentence_map[sent_idx_counter] = sent
                id_score_map[sent_idx_counter] = score
                sent_id_date_map[sent_idx_counter] = date
                date_id_map[date_idx].append(sent_idx_counter)
                id_tok_count_map[sent_idx_counter] = len(sent)

                sent_idx_counter += 1

            cluster_idx_counter += 1

    constraints = [SimilarityConstraint(id_sentence_map), MaxDateCountConstraint(parameters.max_date_count, sent_id_date_map)]

    for date_id, member_ids in date_id_map.items():
        if parameters.max_token_count is not None:
            constraints.append(SubsetKnapsackConstraint(parameters.max_token_count, id_tok_count_map, member_ids))
        else:
            constraints.append(ConstantSizeSubsetKnapsackConstraint(parameters.max_date_sent_count, member_ids))

    if disallow_cluster_repetition:
        constraints.append(ClusterMembershipConstraint(id_cluster_map))

    opt = SubModularOptimizer(
        [ConstantScoreFactor(id_score_map)],
        constraints
    )

    sent_ids = opt.run(range(sent_idx_counter), id_sentence_map)

    selected_tl_sentences = defaultdict(list)
    for sent_id in sent_ids:

        date = sent_id_date_map[sent_id]
        sent = " ".join([tok for tok, pos in id_sentence_map[sent_id]])
        selected_tl_sentences[date].append(sent)

    return selected_tl_sentences


def select_tl_sentences_submod(per_date_cluster_candidates, doc_sents, parameters, tr_scores=None, disallow_cluster_repetition=True, use_small_clusters=False, use_bucket_coverage=False, use_redundancy=True, use_bucket_redundancy=False):
    id_sentence_map = {}
    id_cluster_map = {}
    date_id_map = defaultdict(list)
    id_date_map = {}
    id_score_map = {}
    id_tok_count_map = {}
    sent_id_date_map = {}

    cluster_id_map = defaultdict(list)

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

                cluster_id_map[cluster_idx_counter].append(sent_idx_counter)

                sent_idx_counter += 1

            cluster_idx_counter += 1

    print("Selecting from {} sentences".format(sent_idx_counter))

    constraints = []

    for date_id, member_ids in date_id_map.items():
        if parameters.max_token_count is not None:
            constraints.append(SubsetKnapsackConstraint(parameters.max_token_count, id_tok_count_map, member_ids))
        else:
            constraints.append(ConstantSizeSubsetKnapsackConstraint(parameters.max_date_sent_count, member_ids))

    constraints.append(MaxDateCountConstraint(parameters.max_date_count, sent_id_date_map))

    if disallow_cluster_repetition:
        constraints.append(ClusterMembershipConstraint(id_cluster_map))

    factors = []

    if use_redundancy:
        print("Initializing redudancy")
        num_clusters = None
        if use_small_clusters:
            num_clusters = int(sent_idx_counter // 50)
        else:
            num_clusters = 2 * parameters.max_date_sent_count * parameters.max_date_count

        kmeans_redundancy_factor = RedundancyFactor.from_sentences(
            dict(id_score_map),
            id_sentence_map,
            cluster_id_map,
            num_clusters=min(max(sent_idx_counter // 5, 2), num_clusters))
        factors.append(kmeans_redundancy_factor)

    if use_bucket_coverage:
        factors.append(BucketedCoverageFactor.from_sentences(doc_sents, id_sentence_map, sent_id_date_map))

    if use_bucket_redundancy:
        factors.append(BucketedRedundancyFactor.from_sentences(dict(id_score_map), doc_sents, id_sentence_map))

    opt = SubModularOptimizer(
        factors,
        constraints)

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


class SentenceScorer:
    def __init__(self, config):
        self.lm = KenLMLanguageModel.from_file("langmodel20k_vp_3.bin")

        self.use_lm = False
        self.use_local_informativeness = False
        self.use_cluster_size = False

        self.use_lm = config.get("use_lm", True)
        self.lm_aggregator = config.get("lm_aggregator", "avg")

        self.use_local_informativeness = config.get("use_local_informativeness", True)
        assert "use_local_info" not in config
        self.use_cluster_size = config.get("use_cluster_size", False)

        self.use_global_informativeness = config.get("use_global_informativeness", False)
        self.use_temporalized_informativeness = config.get("use_temporalized_informativeness", False)
        self.use_per_date_informativeness = config.get("use_per_date_informativeness", False)
        self.use_date_frequency = config.get("use_date_frequency", False)

        self.use_path_weight = config.get("use_path_weight", False)

        self.temporalized_informativeness_cache_dir = config.get("temporalized_informativeness_cache_dir")

        self.use_rel_frequency = config.get("use_rel_frequency", False)

        self.use_length = config.get("use_length", False)

        if self.use_path_weight:
            self.use_length = False

        if config.get("global_only", False):
            self.use_lm = False
            self.use_local_informativeness = False
            self.use_cluster_size = True

    def prepare(self, corpus):
        if self.use_date_frequency:
            self.relative_date_frequencies = compute_relative_date_frequencies(corpus)

        if self.use_global_informativeness:
            self.global_tr_scores = calculate_keyword_text_rank([sent.as_token_tuple_sequence("form", "pos") for doc in corpus for sent in doc])

        self.corpus_name = corpus.name

    def compute_date_tr(self, clusters):
        sentences_per_date = defaultdict(list)

        for cluster, date in clusters:
            if date is not None:
                sentences_per_date[date].extend(cluster)
        logger.debug("Must recompute temp tr")
        per_date_tr_scores = {}

        for idx, (date, sentences) in enumerate(sentences_per_date.items()):
            trs = calculate_keyword_text_rank([s.as_token_tuple_sequence("form", "pos") for s in sentences])
            per_date_tr_scores[date] = trs

        return per_date_tr_scores

    def prepare_for_clusters(self, cluster_promise):
        clusters = cluster_promise.get()
        self.max_cluster_size = max(map(lambda c: len(c[0]), clusters))

        date_tr_promise = cluster_promise.chain(self.compute_date_tr, key="date_tr")

        sentences_per_date = defaultdict(list)

        for cluster, date in clusters:
            if date is not None:
                sentences_per_date[date].extend(cluster)

        if self.use_per_date_informativeness or self.use_temporalized_informativeness:
                self.per_date_tr_scores = date_tr_promise.get()

        if self.use_temporalized_informativeness:
            self.per_date_temporalized_tr_scores = compute_temp_tr(self.per_date_tr_scores, sents=sentences_per_date, use_cl_size_reweighting=True)

    def score_function_for_cluster(self, cluster, cluster_date):
        if self.use_local_informativeness:
            local_tr_scores = calculate_keyword_text_rank([s.as_token_tuple_sequence("form", "pos") for s in cluster])

        def calc_local_informativeness(sent):
            local_informativeness_score = 0
            for token in set(sent):
                local_informativeness_score += local_tr_scores.get(token, 0)

            return local_informativeness_score

        def calc_date_informativeness(sent):
            local_informativeness_score = 0
            scores = self.per_date_tr_scores[cluster_date]

            if len(scores.values()) == 0:
                max_score = 0
                min_score = 0
            else:
                max_score = max(scores.values())
                min_score = min(scores.values())

            if max_score - min_score == 0:
                return 0

            for token in set(sent):
                score = self.per_date_tr_scores[cluster_date].get(token, 0)

                if score == 0:
                    continue

                local_informativeness_score += (score - min_score) / (max_score - min_score)

            return local_informativeness_score

        def calc_temporalized_informativeness(sent):
            local_informativeness_score = 0
            for token in set(sent):
                local_informativeness_score += self.per_date_temporalized_tr_scores[cluster_date].get(token, 0)

            return local_informativeness_score

        def calc_global_informativeness(sent):
            local_informativeness_score = 0
            for token in set(sent):
                local_informativeness_score += self.global_tr_scores.get(token, 0)

            return local_informativeness_score

        def sfunc_mult(sidx, sent, info):
            score_info = {}

            score = 1

            if self.use_local_informativeness:
                local_informativeness_score = calc_local_informativeness(sent)
                score *= local_informativeness_score
                score_info["local_informativeness_score"] = local_informativeness_score

            if self.use_per_date_informativeness:
                date_informativeness = calc_date_informativeness(sent)
                score *= date_informativeness
                score_info["date_informativeness"] = date_informativeness

            if self.use_temporalized_informativeness:
                temp_informativeness = calc_temporalized_informativeness(sent)
                score *= temp_informativeness
                score_info["temp_informativeness"] = temp_informativeness

            if self.use_global_informativeness:
                global_informativeness = calc_global_informativeness(sent)
                score *= global_informativeness
                score_info["global_informativeness"] = global_informativeness

            if self.use_lm:
                if self.lm_aggregator == "avg":
                    lm_score = 1 / (1.0 - self.lm.estimate_sent_log_proba(sent))
                elif self.lm_aggregator == "min-avg":
                    min_probas = sorted(self.lm.estimate_full_sent_log_probas(sent)[:5])
                    lm_score = 1 / (1.0 - sum(min_probas) / len(min_probas))

                score *= lm_score
                score_info["lm_score"] = lm_score

            if self.use_cluster_size:
                cluster_size_score = len(cluster) / self.max_cluster_size
                score *= cluster_size_score
                score_info["cluster_size_score"] = cluster_size_score

            if self.use_date_frequency:
                least_frequency = min(self.relative_date_frequencies.values())

                score *= self.relative_date_frequencies.get(cluster_date, least_frequency)
                score_info["relative_date_frequency"] = self.relative_date_frequencies.get(cluster_date, least_frequency)

            if self.use_rel_frequency:
                score *= info["rel_frequency"]
                score_info["rel_frequency"] = info["rel_frequency"]

            if self.use_path_weight:
                score *= 1.0 / (1 + info["weight"])
                score_info["path_weight"] = info["weight"]

            if self.use_length:
                score *= 1.0 / len(sent)

            return score, score_info

        return sfunc_mult


class ROUGESenteneScorer:
    def compute_rouge_scores(self, corpus, per_date_clusters, per_date_candidates, timelines):
        per_timeline_rouge_scored_per_date_clusters = None
        rouge_cache_basepath = self.config.get("rouge_cache_path")
        if rouge_cache_basepath is not None:
            rouge_cache_path = os.path.join(rouge_cache_basepath, corpus.name.replace("/", "_") + ".pkl")
            logger.debug("Reading from ROUGE cache path {}".format(rouge_cache_path))
            if os.path.isfile(rouge_cache_path):
                with open(rouge_cache_path, "rb") as f:
                    per_timeline_rouge_scored_per_date_clusters = pickle.load(f)

        if per_timeline_rouge_scored_per_date_clusters is None:
            per_timeline_rouge_scored_per_date_clusters = {}
            for tlname, timeline in timelines:
                logger.debug("Computing Rouge")
                per_timeline_rouge_scored_per_date_clusters[tlname] = compute_rouge_for_clusters(per_date_candidates, [timeline])

        if rouge_cache_basepath is not None:
            rouge_cache_path = os.path.join(rouge_cache_basepath, corpus.name.replace("/", "_") + ".pkl")
            if not os.path.isfile(rouge_cache_path):
                if not os.path.isdir(rouge_cache_basepath):
                    os.makedirs(rouge_cache_basepath)

                with open(rouge_cache_path, "wb") as f_out:
                    pickle.dump(per_timeline_rouge_scored_per_date_clusters, f_out)

        return per_timeline_rouge_scored_per_date_clusters


def safe_max(l):
    list_ = list(l)
    if len(list_) == 0:
        return 0
    else:
        return max(list_)


class GraphCandidateGenerator:
    def __init__(self, config):
        self.use_weighting = config.get("use_weighting", True)
        self.maxlen = config.get("sent_maxlen", None)
        self.use_dep_filtering = config.get("use_dep_filtering", False)
        self.force_abstractive = config.get("force_abstractive", False)

    def prepare(self, corpus):
        self.tfidf_model = TfidfVectorizer()
        self.tfidf_model.fit(list(map(lambda s: s.as_tokenized_string(), corpus.sentences)))

    def generate_all_candidates(self, clusters):
        for cl in clusters:
            for s in cl:
                if s.dependency_tree is None:
                    print(cl)
        inputs = [[
            (
                list(sent.as_token_tuple_sequence("form_lowercase", "pos")),
                sent.dependency_tree.as_head_idx_sequence()
            ) for sent in cluster] for cluster in clusters]
#        inputs = [[sent for sent in cluster] for cluster in clusters]

        batches = []

        batch_cursor = 0
        batch_size = 50

        while batch_cursor < len(clusters):
            batches.append(inputs[batch_cursor:batch_cursor + batch_size])
            batch_cursor += batch_size

        pool = get_context('spawn').Pool(cpu_count() - 1)

        for batch_result in pool.imap(self.process_batch, batches):
            for result in batch_result:
                yield result

        pool.close()
        pool.join()

    def process_batch(self, batch):
        results = []
        for cluster in batch:
            results.append(self.generate_candidates(cluster))

        return results

    def generate_candidates(self, cluster):
        all_candidates_and_info = []

        compressor = SentenceCompressionGraph(STOPWORDS)
        for s, dep_data in cluster:
            compressor.add_sentence(s, dep_data)

        cluster_vectors = self.tfidf_model.transform(list(map(lambda s: " ".join([t[0] for t in s[0]]), cluster)))

        def check_closeness(sent):
            if not self.force_abstractive:
                return True

            sent_vec = self.tfidf_model.transform([" ".join(map(lambda x: x[0], sent))])
            sims = cosine_similarity(sent_vec, cluster_vectors)

            return all(sims[0,:] <= 0.8)

        for proposed_sent, path_info in compressor.generate_compression_candidates(
                filterfunc=check_closeness,
                use_weighting=self.use_weighting,
                maxlen=self.maxlen,
                return_weight=True,
                use_dep_filtering=self.use_dep_filtering):

            all_candidates_and_info.append((proposed_sent, {"weight": path_info["weight"], "rel_frequency": path_info["avg_rel_frequency"]}))

        return all_candidates_and_info


class MayorityClusterDater:
    def __init__(self, config):
        pass

    def date_cluster(self, cluster):
        referenced_dates = Counter()

        for sent in cluster:
            referenced_dates.update(sent.exact_date_references)
            referenced_dates.update([sent.document.dct_tag])

        if len(referenced_dates) == 0:
            return None
        cluster_tag, _ = referenced_dates.most_common(1)[0]

        try:
            cluster_date = datetime.date(cluster_tag.year, cluster_tag.month, cluster_tag.day)
        except ValueError:
            return None

        return cluster_date


class DCTClusterDater:
    def __init__(self, config):
        pass

    def date_cluster(self, cluster):
        referenced_dates = Counter()

        for sent in cluster:
                referenced_dates.update([sent.document.dct_tag])

        if len(referenced_dates) == 0:
            return None

        cluster_tag, _ = referenced_dates.most_common(1)[0]

        try:
            cluster_date = datetime.date(cluster_tag.year, cluster_tag.month, cluster_tag.day)
        except ValueError:
            return None

        return cluster_date


class APClusterer:
    def __init__(self, config):
        self.include_uncertain_date_edges = config.get("include_uncertain_date_edges", True)
        self.predicted_tag_only = config.get("predicted_tag_only", False)
        self.untangle = config.get("untangle", False)
        self.post_prune = config.get("post_prune", False)

    def cluster_corpus(self, corpus):
        clusters = cluster_sentences_ap(
            corpus.sentences,
            include_uncertain_date_edges=self.include_uncertain_date_edges,
            predicted_tag_only=self.predicted_tag_only,
            untangle_multi_date_sentences=self.untangle,
            post_prune=self.post_prune)

        return [Cluster(c) for c in clusters]


class GlobalSubModularSentenceSelector:
    def __init__(self, config):
        self.use_small_clusters = config.get("use_small_clusters", False)
        self.disallow_cluster_repetition = config.get("disallow_cluster_repetition", True)

        self.use_bucket_coverage = config.get("use_bucket_coverage", False)
        self.use_redundancy = config.get("use_redundancy", True)

        self.use_bucket_redundancy = config.get("use_bucket_redundancy", False)

    def prepare(self, corpus):
        self.corpus = corpus

    def select_sentences_from_clusters(self, per_date_clusters, parameters, tr_scores):
        return select_tl_sentences_submod(per_date_clusters, self.corpus.sentences, parameters, tr_scores=tr_scores, use_small_clusters=self.use_small_clusters, disallow_cluster_repetition=self.disallow_cluster_repetition, use_bucket_coverage=self.use_bucket_coverage, use_redundancy=self.use_redundancy, use_bucket_redundancy=self.use_bucket_redundancy)


class GreedySentenceSelector:
    def __init__(self, config):
        self.disallow_cluster_repetition = config.get("disallow_cluster_repetition", False)

    def prepare(self, corpus):
        self.corpus = corpus

    def select_sentences_from_clusters(self, per_date_clusters, parameters, tr_scores):
        return select_tl_sentences_greedy(per_date_clusters, self.corpus.sentences, parameters, disallow_cluster_repetition=self.disallow_cluster_repetition)


class IdentityCandidateGenerator:
    def __init__(self, config):
        pass

    def prepare(self, corpus):
        pass

    def generate_candidates(self, cluster):
        return [(s.as_token_tuple_sequence("form", "pos"), None) for s in cluster]


class ROUGEOracleScorer:
    def __init__(self, config):
        self.alignment_method = config.get("alignment_method", "agree")
        self.scoring_method = config.get("scoring_method", "f1")

    def prepare_for_clusters(self, clusters):
        pass

    def score_clusters_for_timeline(self, per_date_cluster_candidates, timeline):
        scores = list(compute_rouge_for_clusters(per_date_cluster_candidates, [timeline], alignment_method=self.alignment_method, scoring_method=self.scoring_method).items())
        return scores


def compute_scores_from_rouge_counts(scores):
    if scores["rouge_1_p_count"] > 0:
        prec = scores["rouge_1_h_count"] / scores["rouge_1_p_count"]
    else:
        prec = 0.0
    if scores["rouge_1_m_count"] > 0:
        rec = scores["rouge_1_h_count"] / scores["rouge_1_m_count"]
    else:
        rec = 0.0
    if prec + rec > 0:
        score = (prec * rec) / (prec + rec)
    else:
        score = 0

    return prec, rec, score


def compute_rouge_for_clusters(per_date_cluster_candidates, timelines, combine_method="avg", alignment_method="agree", scoring_method="f1"):
    scored_per_date_cluster_candidates = {}
    rouge_eval = RougeReimplementation()

    for date, clusters in per_date_cluster_candidates:
        all_references = []
        for timeline in timelines:
            score_modifier = 1
            if alignment_method == "agree":
                reference = {"A": [s.split() for s in timeline[date]]}
            elif alignment_method == "concat":
                reference = {"A": [s.split() for s in timeline[date] for date in timeline.get_dates()]}
            elif alignment_method == "closest":
                closest_date = min(timeline.get_dates(), key=lambda d: abs(d - date))
                reference = {"A": [s.split() for s in timeline[closest_date]]}

                score_modifier = 1.0 / (abs(closest_date - date).days + 1)
            else:
                raise ValueError("Invalid scoring method")

            all_references.append((timeline, reference, score_modifier))

        scored_clusters = []
        for cluster in clusters:
            scored_sentences = []
            for sent, info in cluster:
                all_scores = []
                for timeline, reference, score_modifier in all_references:
                    if date in timeline.get_dates():
                        scores = rouge_eval.score_summary([list(map(fst, sent))], reference)
                        prec, rec, f1 = compute_scores_from_rouge_counts(scores)

                        if scoring_method == "f1":
                            score = f1
                        elif scoring_method == "prec":
                            score = prec
                        elif scoring_method == "rec":
                            score = rec
                        else:
                            raise ValueError("Invalid scoring method")
                    else:
                        score = 0

                    all_scores.append(score * score_modifier + 1)

                if combine_method == "avg":
                    final_score = sum(all_scores) / len(all_scores)
                else:
                    raise ValueError("Unknown combine method {!r}".format(combine_method))

                scored_sentences.append((sent, final_score))
            scored_clusters.append(scored_sentences)
        scored_per_date_cluster_candidates[date] = scored_clusters

    return scored_per_date_cluster_candidates


class GloballyClusteredSentenceCompressionTimelineGenerator:
    def clusterer_from_config(self, config):
        method = config["method"]

        self.clustering_method = method

        if config.get("untangle") is True:
            self.clustering_method += "untangle"

        if config.get("post_prune") is True:
            self.clustering_method += "prune"

        if method == "ap":
            return APClusterer(config)
        else:
            raise ValueError("Method {!r} not recognized".format(method))

    def selector_from_config(self, config):
        method = config["method"]

        if method == "global_submod":
            return GlobalSubModularSentenceSelector(config)
        elif method == "greedy":
            return GreedySentenceSelector(config)
        else:
            raise ValueError("Method {!r} not recognized".format(method))

    def generator_from_config(self, config):
        method = config.get("method", "graph")

        self.generation_method = method

        if config.get("use_dep_filtering"):
            self.generation_method += "-dep-filtered"

        if config.get("force_abstractive"):
            self.generation_method += "-full-abs"

        if method == "graph":
            return GraphCandidateGenerator(config)
        else:
            raise ValueError("Method {!r} not recognized".format(method))

    def scorer_from_config(self, config):
        method = config.get("method", "default")
        if method == "default":
            return SentenceScorer(config)
        elif method == "rouge_oracle":
            return ROUGEOracleScorer(config)
        else:
            raise ValueError("Method {!r} not recognized".format(method))

    def date_selector_from_config(self, config):
        method = config.get("method")

        if method is None:
            raise ValueError("Date selection method must be specified")

        elif method == "dateref_freq":
            return DateFreqeuencyDateSelector()
        else:
            raise ValueError("Method {!r} not recognized".format(method))

    def cluster_dater_from_config(self, config):
        method = config.get("method", "mayority")

        print(method)

        if method == "mayority":
            return MayorityClusterDater(config)
        elif method == "dct":
            return DCTClusterDater(config)

    def __init__(self, config):
        self.config = config

        self.clusterer = self.clusterer_from_config(config["clustering"])
        self.scorer = self.scorer_from_config(config.get("scoring", {}))
        self.generator = self.generator_from_config(config.get("generation", {}))
        self.cluster_dater = self.cluster_dater_from_config(config.get("dating", {}))
        self.sentence_selector = self.selector_from_config(config["selection"])

        self.use_cooc_filtering = config.get("use_cooc_filtering", False)

        self.date_selector = None
        date_sel_config = config.get("date_selection")

        if date_sel_config is not None:
            self.date_selector = self.date_selector_from_config(date_sel_config)

        self.min_cluster_size = config.get("min_cluster_size", 0)

    def run_scoring_cv_train_mode(self, corpora_and_timelines):
        preprocessed_corpora = {}

        for corpus, timelines in corpora_and_timelines:
            self.generator.prepare(corpus)
            clusters = self.create_clusters(corpus)
            cluster_candidates = self.generate_candidates_for_clusters(corpus, clusters)

            per_date_cluster_candidates = defaultdict(list)
            per_date_clusters = defaultdict(list)
            for cluster, candidates_and_info in zip(clusters, cluster_candidates):
                cluster_date = self.date_cluster(cluster)

                if cluster_date is None:
                    continue

                per_date_clusters[cluster_date].append(cluster)
                per_date_cluster_candidates[cluster_date].append(candidates_and_info)

            per_date_cluster_candidates = list(per_date_cluster_candidates.items())

            preprocessed_corpora[corpus] = (per_date_clusters, per_date_cluster_candidates, timelines)

        parameters = {}

        for corpus, timelines in corpora_and_timelines:
            training_corpora = [(c, preprocessed_corpora[c][0], preprocessed_corpora[c][1], preprocessed_corpora[c][2]) for c, _ in corpora_and_timelines if c != corpus]

            parameters[corpus.name] = self.scorer.train(training_corpora)

        return parameters

    def generate_corpus_statistics(self, corpus):
        self.scorer.prepare(corpus)
        self.generator.prepare(corpus)
        self.sentence_selector.prepare(corpus)
        clusters = self.create_clusters(corpus)
        dated_clusters = list(((cluster, self.date_cluster(cluster)) for cluster in clusters))
        cluster_candidates = self.generate_candidates_for_clusters(corpus, clusters)

        total_cluster_count = len(clusters)
        total_candidate_count = 0

        for (cluster, cluster_date), candidates_and_info in sorted(zip(dated_clusters, cluster_candidates), key=lambda x: len(x[0][0]), reverse=True):
            total_candidate_count += len(candidates_and_info)

        return {
            "candidate_count": total_candidate_count,
            "cluster_count": total_cluster_count,
            "sentence_count": len(corpus.sentences),
            "doc_count": corpus.num_documents
        }

    def generate_date_docs_cluster_stats(self, corpus):
        self.generator.prepare(corpus)
        clusters = self.create_clusters(corpus)
        dated_clusters = list(((cluster, self.date_cluster(cluster)) for cluster in clusters))
        cluster_candidates = self.generate_candidates_for_clusters(corpus, clusters)

        candidate_counts_per_date = Counter()
        for ((cluster, cluster_date), candidates_and_info) in zip(dated_clusters, cluster_candidates):
            candidate_counts_per_date[cluster_date] += len(candidates_and_info)

        samples = []
        for date in set(corpus.iter_dates()) | candidate_counts_per_date.keys():
            n_docs = len(corpus.per_date_documents[date])
            n_candidates = candidate_counts_per_date[date]
            samples.append((n_docs, n_candidates))

        return samples

    def get_promises(self, corpus):
        corpus_promise = ConstantPromise(corpus, corpus.name, CacheManager())

        def to_cluster(cl):
            if isinstance(cl[0], Cluster):
                return cl
            else:
                return [Cluster(c) for c in cl]

        cluster_promise = corpus_promise.chain(self.clusterer.cluster_corpus, key=self.clustering_method, transform=to_cluster)

        cluster_candidates = cluster_promise.chain(lambda clusters: self.generate_candidates_for_clusters(corpus, clusters), key=self.generation_method).get()

        dated_clusters_promise = cluster_promise.chain(lambda clusters: list(((cluster, self.date_cluster(cluster)) for cluster in clusters)), key="dated", cacheable=False)

        return corpus_promise, cluster_promise, dated_clusters_promise, cluster_candidates

    def date_cluster(self, cluster):
        if cluster.date is None:
            return self.cluster_dater.date_cluster(cluster)
        else:
            try:
                cluster_date = datetime.date(cluster.date.year, cluster.date.month, cluster.date.day)
            except ValueError:
                return None

            return cluster_date

    def generate_timelines(self, corpus, all_parameters, query_words=None, reference_timelines=None, debug_identifier=None, num_selection_runs=None):
        self.scorer.prepare(corpus)
        self.generator.prepare(corpus)
        self.sentence_selector.prepare(corpus)

        corpus_promise, cluster_promise, dated_clusters_promise, cluster_candidates = self.get_promises(corpus)
        clusters = cluster_promise.get()

        dated_clusters = dated_clusters_promise.get()

        per_date_cluster_candidates = defaultdict(list)

        self.scorer.prepare_for_clusters(dated_clusters_promise)

        for cl_rank, ((cluster, cluster_date), candidates_and_info) in enumerate(sorted(zip(dated_clusters, cluster_candidates), key=lambda x: len(x[0][0]), reverse=True)):
            if cluster_date is None:
                continue

            cluster_candidates = []
            if hasattr(self.scorer, "score_function_for_cluster"):
                score_func = self.scorer.score_function_for_cluster(cluster, cluster_date)
                if self.use_cooc_filtering:
                    candidates_and_info = filter_canidates_by_integrity(cluster, candidates_and_info)
                for sidx, (candidate, info) in enumerate(candidates_and_info):
                    score, info = score_func(sidx, candidate, info)

                    forms = set(w[0] for w in candidate)
                    if query_words is None or any(qw in forms for qw in query_words):
                        cluster_candidates.append((candidate, score, info))

                cluster_candidates = list(
                    map(lambda x: (x[0], x[1]), cluster_candidates)
                )

            else:
                cluster_candidates = candidates_and_info

            if len(cluster) < self.min_cluster_size:
                continue

            per_date_cluster_candidates[cluster_date].append(cluster_candidates)

        per_date_cluster_candidates = list(per_date_cluster_candidates.items())

        global_tr_scores = None

        num_selection_runs_ = 1
        if num_selection_runs is not None:
            num_selection_runs_ = num_selection_runs

        if hasattr(self.scorer, "score_clusters_for_timelines") and reference_timelines is not None:
            per_date_cluster_candidates = self.scorer.score_clusters_for_timelines(per_date_cluster_candidates, reference_timelines)

        per_run_timelines = []
        for n in range(num_selection_runs_):
            all_timelines = []
            for timeline_idx, parameters in enumerate(all_parameters):
                local_per_date_cluster_candidates = per_date_cluster_candidates
                if self.date_selector is not None:
                    selected_dates = self.date_selector.select_dates(corpus, parameters)

                    local_per_date_cluster_candidates = [
                        (date, date_clusters) for date, date_clusters in per_date_cluster_candidates if date in selected_dates]

                if hasattr(self.scorer, "score_clusters_for_timeline") and reference_timelines is not None:
                    # For oracle summarization
                    local_per_date_cluster_candidates = self.scorer.score_clusters_for_timeline(local_per_date_cluster_candidates, reference_timelines[timeline_idx])

                checking_outfile_name = None
                if debug_identifier is not None:
                    checking_outfile_name = "candidate-debug-info-{}-{}".format(debug_identifier, timeline_idx)

                timeline = self.sentence_selector.select_sentences_from_clusters(local_per_date_cluster_candidates, parameters, global_tr_scores)

                if checking_outfile_name is not None:
                    all_sents = []
                    for date, clusters in per_date_cluster_candidates:
                        for candidates in clusters:
                            for candidate, score in candidates:
                                cand_str = " ".join([tok for tok, pos in candidate])
                                all_sents.append("\t".join((cand_str, str(date), str(cand_str in timeline.get(date, [])))))
                    all_sents.sort()

                    with open(checking_outfile_name, "w") as f_out:
                        for line in all_sents:
                            f_out.write(line)
                            f_out.write("\n")

                if self.date_selector is not None:
                    for date in selected_dates:
                        if date not in timeline:
                            timeline[date] = ""

                all_timelines.append(Timeline(timeline))

            per_run_timelines.append(all_timelines)

        if num_selection_runs is None:
            return per_run_timelines[0]
        else:
            return per_run_timelines

    def create_clusters(self, corpus):
        base_cache_path = self.config.get("cluster_cache_path")
        base_cache_path = None
        print("WARNING! Deprecated caching method")
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
            clusters = [cluster for _, cluster in sorted(read_clusters(cache_path, corpus).items())]

        clusters = [Cluster(cluster) for cluster in clusters]

        return clusters

    def generate_candidates_for_clusters(self, corpus, clusters):
        logger.debug("Processing {} clusters".format(len(clusters)))

        base_cache_path = self.config.get("candidate_cache_dir")
        base_cache_path = None
        print("WARNING! Deprecated caching method")

        outfile = None

        if base_cache_path is not None:
            pickle_cache_path = os.path.join(base_cache_path, (corpus.name + ".txt.pkl").replace("/", "_"))

            if os.path.isfile(pickle_cache_path):
                with open(pickle_cache_path, "rb") as f:
                    return pickle.load(f)

            cache_path = os.path.join(base_cache_path, (corpus.name + ".txt").replace("/", "_"))

            if os.path.isfile(cache_path):
                candidates = list(read_candidate_file(cache_path, clusters))
                return candidates
            else:
                if not os.path.isdir(base_cache_path):
                    os.makedirs(base_cache_path)
                outfile = open(cache_path, "w")

        all_candidates_and_info = []

        import time
        start = time.time()

        if hasattr(self.generator, "generate_all_candidates"):
            for cluster_idx, (candidates_and_info, cluster) in enumerate(zip(self.generator.generate_all_candidates(clusters), clusters)):
                if (cluster_idx + 1) % 100 == 0:
                    logger.debug("Generating sentences for cluster {} of {} ({})".format(cluster_idx, len(clusters), datetime.timedelta(seconds=time.time() - start)))

                if outfile is not None:
                    self._write_candidate_set(outfile, candidates_and_info)

                all_candidates_and_info.append(candidates_and_info)
        else:
            for cluster_idx, cluster in enumerate(clusters):
                if (cluster_idx + 1) % 100 == 0:
                    logger.debug("Generating sentences for cluster {} of {} ({})".format(cluster_idx, len(clusters), datetime.timedelta(seconds=time.time() - start)))

                candidates_and_info = self.generator.generate_candidates(cluster)

                if outfile is not None:
                    self._write_candidate_set(outfile, candidates_and_info)

                all_candidates_and_info.append(candidates_and_info)

        if outfile is not None:
            outfile.close()

        return all_candidates_and_info

    def _write_candidate_set(self, f_out, candidates_and_info):
        for sentence, info in candidates_and_info:
            tokens, pos = zip(*sentence)
            f_out.write("{}\n".format(" ".join(map(lambda t: quote(t), tokens))))
            f_out.write("{}\n".format(" ".join(pos)))
            f_out.write("{}\n".format(json.dumps(info)))
        f_out.write("\n")


def read_candidate_file(fname, clusters):
    with open(fname) as f:
        for cluster in clusters:
            tok_cache = {}

            sentences = []

            first_subset_iter = True
            while True:
                try:
                    token_line = next(f).strip()
                except EOFError:
                    break
                first_subset_iter = False

                if len(token_line) == 0:
                    break

                tokens = [unquote(tok) for tok in token_line.split(" ")]
                pos_line = next(f).strip()
                pos = pos_line.split()

                sentence = list(map(lambda x: tok_cache.setdefault(x, x), zip(tokens, pos)))

                info = json.loads(next(f))

                sentences.append((sentence, info))

                first_subset_iter = False

            if first_subset_iter:
                raise RuntimeError("Candidate file corrupted")

            yield sentences


class DateFreqeuencyDateSelector:
    def __init__(self):
        pass

    def select_dates(self, corpus, parameters):
        date_freqs = compute_relative_date_frequencies(corpus)

        selected_dates = set()

        for date, freq in sorted(date_freqs.items(), key=lambda i: i[1], reverse=True):
            if date < parameters.first_date or date > parameters.last_date:
                continue

            selected_dates.add(date)

            if len(selected_dates) >= parameters.max_date_count:
                break

        return selected_dates


def build_essential_cooc_list(cluster):

    combo_freqs = Counter()
    word_freqs = Counter()

    for sent in cluster:
        words = set(filter(lambda s: s not in "?!.,;", sent.as_token_attr_sequence("form_lowercase")))
        all_combos = list(map(lambda x: tuple(sorted(x)), it.combinations(words, 2)))
        combo_freqs.update(all_combos)
        word_freqs.update(words)

    word_cond_probas = {}

    for (l_word, r_word), combo_freq in combo_freqs.items():
        word_cond_probas[(l_word, r_word)] = combo_freq / word_freqs[r_word]
        word_cond_probas[(r_word, l_word)] = combo_freq / word_freqs[l_word]

    essential_combos = defaultdict(set)

    for (l_word, r_word), prop in word_cond_probas.items():
        if prop >= 1.0 and word_freqs[r_word] > 1:
            essential_combos[r_word].add(l_word)

    return essential_combos


def filter_canidates_by_integrity(cluster, candidates):
    essential_cooc = build_essential_cooc_list(cluster)

    for cand, info in candidates:
        if check_candidate_integrity(essential_cooc, cand):
            yield cand, info


def check_candidate_integrity(essential_cooc, candidate):
    forms = set([w[0] for w in candidate])
    for form in forms:
        for required_word in essential_cooc.get(form, []):
            if required_word not in forms:
                return False

    return True
