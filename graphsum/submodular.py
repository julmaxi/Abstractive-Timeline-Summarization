import math
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter

import logging

logger = logging.getLogger(__name__)

class RedundancyFactor:
    @classmethod
    def from_sentences(cls, rewards, id_sentence_map, num_clusters=None):
        model = TfidfVectorizer(stop_words="english")
        sorted_sent_ids = sorted(id_sentence_map.keys())

        tf_idf = model.fit_transform(map(lambda sid: " ".join(
            map(lambda t: t[0], id_sentence_map[sid])), sorted_sent_ids))

        if num_clusters is None:
            num_clusters = max(len(id_sentence_map) // 5, 2)

        clustering = MiniBatchKMeans(n_clusters=num_clusters).fit_predict(tf_idf)

        clustering = [0] * len(sorted_sent_ids)

        sent_partitions = {}

        for cl_id, sent_id in zip(clustering, sorted_sent_ids):
            sent_partitions[sent_id] = cl_id

        return RedundancyFactor(rewards, sent_partitions)

    def __init__(self, rewards, sent_partitions, normalize=True):
        self.rewards = rewards
        self.sent_partitions = sent_partitions

        #self.all_partitions = set(sent_partitions.values())

        if normalize:
            self._normalize_rewards()

        self.reward_sums = dict((partition_id, 0) for partition_id in sent_partitions)
        self.reward_sqrts = dict((partition_id, 0) for partition_id in sent_partitions)

    def _normalize_rewards(self):
        max_reward = 0
        for reward in self.rewards.values():
            max_reward = max(reward, max_reward)

        for sid, reward in list(self.rewards.items()):
            self.rewards[sid] = reward / max_reward

    def update_scores(self, new_sentence):
        new_sent_partition_id = self.sent_partitions[new_sentence]

        self.reward_sums[new_sent_partition_id] += self.rewards[new_sentence]
        self.reward_sqrts[new_sent_partition_id] = math.sqrt(self.reward_sums[new_sent_partition_id])

    def find_score_delta(self, sent):
        sent_partition_id = self.sent_partitions[sent]
        delta_s = math.sqrt(self.reward_sums[sent_partition_id] + self.rewards[sent]) - self.reward_sqrts[sent_partition_id]

        return delta_s


class CoverageFactor:
    @classmethod
    def from_sentences(cls, doc_sents, summary_sents):
        model = TfidfVectorizer(stop_words="english", max_features=1024, min_df=5)
        doc_tf_idf = model.fit_transform(map(lambda s: " ".join(
            map(lambda t: t[0], s)), doc_sents))
        summ_tf_idf = model.transform(map(lambda s: " ".join(
            map(lambda t: t[0], s)), summary_sents))

        #cosine = cosine_similarity(summ_tf_idf, doc_tf_idf)

        overlaps = {}

        for idx, summ_tf_idf_vec in enumerate(summ_tf_idf):
            cosine = cosine_similarity(summ_tf_idf_vec, doc_tf_idf)
            overlaps[idx] = np.sum(cosine) / doc_tf_idf.shape[0]

        return CoverageFactor(overlaps)

    def __init__(self, overlaps, normalize=True):
        self.overlaps = overlaps
        #self.overlaps = {}
#
        #for sent_id, sent_row in enumerate(similarity_matrix):
        #    self.overlaps[sent_id] = sum(sent_row) / similarity_matrix.shape[1]

        if normalize:
            self._normalize_scores()

        self.current_overlap_score = 0.0

    def _normalize_scores(self):
        max_overlap = 0

        for overlap in self.overlaps.values():
            max_overlap = max(max_overlap, overlap)

        for did, overlap in list(self.overlaps.items()):
            self.overlaps[did] = overlap / max_overlap


    def update_scores(self, new_sentence):
        self.current_overlap_score += self.overlaps[new_sentence]

    def find_score_delta(self, sent):
        delta_s = self.overlaps[sent]
        return delta_s


class KnapsackConstraint:
    def __init__(self, knapsack_size, sent_sizes):
        self.knapsack_size = knapsack_size
        self.sent_sizes = sent_sizes
        self.current_size = 0

    def check(self, sent):
        new_size = self.current_size + self.sent_sizes[sent]

        if new_size > self.knapsack_size:
            return False
        else:
            return True

    def update(self, new_sent):
        self.current_size += self.sent_sizes[new_sent]


class SubsetKnapsackConstraint:
    def __init__(self, knapsack_size, sent_sizes, relevant_sents):
        self.knapsack_size = knapsack_size
        self.sent_sizes = sent_sizes
        self.current_size = 0
        self.relevant_sents = set(relevant_sents)

    def check(self, sent):
        if sent not in self.relevant_sents:
            return True
        new_size = self.current_size + self.sent_sizes[sent]

        if new_size > self.knapsack_size:
            return False
        else:
            return True

    def update(self, new_sent):
        if new_sent in self.relevant_sents:
            self.current_size += self.sent_sizes[new_sent]


class ClusterMembershipConstraint:
    def __init__(self, id_cluster_map):
        self.id_cluster_map = id_cluster_map
        self.included_clusters = set()

    def check(self, sent):
        return self.id_cluster_map[sent] not in self.included_clusters

    def update(self, new_sent):
        self.included_clusters.add(self.id_cluster_map[new_sent])


class MaxDateCountConstraint:
    def __init__(self, max_date_count, sent_dates):
        self.selected_dates = set()
        self.sent_dates = sent_dates
        self.max_date_count = max_date_count

    def check(self, sent):
        date = self.sent_dates[sent]
        if date not in self.selected_dates and len(self.selected_dates) >= self.max_date_count:
            return False

        return True

    def update(self, new_sent):
        date = self.sent_dates[new_sent]
        self.selected_dates.add(date)


class SubModularOptimizer:
    def __init__(self, factors, constraints, selection_callback=None):
        self.factors = factors
        self.constraints = constraints

        self.selection_callback = selection_callback

    def run(self, sentences):
        selected_sentences = set()
        filtered_sentences = set()

        while True:
            best_sentence = None
            best_delta_s = 0
            for sent in sentences:
                if sent in filtered_sentences or sent in selected_sentences:
                    continue
                fullfills_constr = self._does_pass_constraints(sent)

                if not fullfills_constr:
                    filtered_sentences.add(sent)
                    continue

                delta_s = self._determine_delta_s(sent)

                if delta_s > best_delta_s:
                    best_sentence = sent
                    best_delta_s = delta_s

            if best_sentence is None:
                break
            else:
                if self.selection_callback is not None:
                    self.selection_callback(best_sentence, best_delta_s)

                logger.info("Selecting sent {} (delta: {})".format(best_sentence, best_delta_s))
                selected_sentences.add(best_sentence)
                for factor in self.factors:
                    factor.update_scores(best_sentence)
                for constraint in self.constraints:
                    constraint.update(best_sentence)

        return selected_sentences

    def _determine_delta_s(self, sent):
        delta_s_sum = 0.0
        for factor in self.factors:
            delta_s_sum += factor.find_score_delta(sent)

        return delta_s_sum

    def _does_pass_constraints(self, sent):
        return all(constr.check(sent) for constr in self.constraints)
