import math
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import time

from collections import Counter

from embeddings import GloveEmbedding
import itertools as it

import logging

from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger(__name__)


def cluster_same_words(sorted_sent_ids, id_sentence_map, num_clusters, **kwargs):
    cluster_keys = {}
    sent_clusters = {}

    curr_cl_id = 0

    for sid in sorted_sent_ids:
        key = tuple(sorted([token for token, _ in id_sentence_map[sid]]))

        if key not in cluster_keys:
            cluster_keys[key] = curr_cl_id
            curr_cl_id += 1

        sent_clusters[sid] = cluster_keys[key]

    return sent_clusters


def cluster_db_scan(sorted_sent_ids, id_sentence_map, num_clusters, cluster_id_map):
    clusters = cluster_id_map

    d_emb = 300
    embeddings = GloveEmbedding('common_crawl_840', d_emb=d_emb, show_progress=True)

    X = fast_sents_to_embeddings(embeddings, 300, [[t[0] for t in id_sentence_map[sid]] for sid in sorted_sent_ids])

    avg_distances = []
    for cl, ids in clusters.items():
        X_cl = X[ids,:]
        dists = np.abs(euclidean_distances(X_cl))
        avg_distances.append(np.sum(dists) / max(1, X_cl.shape[0]**2 - X_cl.shape[0]))

    avg_intra_cl_dist = sum(avg_distances) / len(avg_distances)

    return DBSCAN(avg_intra_cl_dist, n_jobs=-1).fit_predict(X)


def cluster_clustering_kmeans(sorted_sent_ids, id_sentence_map, num_clusters, cluster_id_map):
    clusters = cluster_id_map

    d_emb = 300
    embeddings = GloveEmbedding('common_crawl_840', d_emb=d_emb, show_progress=True)

    X = fast_sents_to_embeddings(embeddings, 300, [[t[0] for t in id_sentence_map[sid]] for sid in sorted_sent_ids])

    avg_distances = []
    X_cls = []
    for cl, ids in sorted(clusters.items()):
        X_cl = X[ids,:]
        dists = np.abs(euclidean_distances(X_cl))
        avg_distances.append(np.sum(dists) / max(1, X_cl.shape[0]**2 - X_cl.shape[0]))

        X_cls.append(X_cl.sum(axis=0))

    X_cls = np.stack(X_cls)

    avg_intra_cl_dist = sum(avg_distances) / len(avg_distances)

    cluster_clusters = KMeans(len(cluster_id_map) // 5).fit_predict(X_cls)

    clusters = [0 for _ in range(len(sorted_sent_ids))]

    for cl_id, cl_cluster_id in zip(sorted(cluster_id_map), cluster_clusters):
        for sent_id in cluster_id_map[cl_id]:
            clusters[sent_id] = cl_cluster_id

    return clusters


def cluster_kmeans_glove(sorted_sent_ids, id_sentence_map, num_clusters):
    d_emb = 300
    embeddings = GloveEmbedding('common_crawl_840', d_emb=d_emb, show_progress=True)

    vecs = np.zeros(shape=(len(sorted_sent_ids), 300))
    for idx, sent_id in enumerate(sorted_sent_ids):
        for token, _ in id_sentence_map[sent_id]:
            vecs[idx] += np.array(embeddings.emb(token.lower(), "zero"))
        vecs[idx] /= len(id_sentence_map[sent_id])

    if num_clusters is None:
        num_clusters = max(len(id_sentence_map) // 25, 2)

    #clusterer = AgglomerativeClustering(n_clusters=num_clusters)

    #clustering = clusterer.fit_predict(vecs)
    clustering = KMeans(n_clusters=num_clusters).fit_predict(vecs)

    return clustering


def random_unit_vecs(d, n_vecs):
    base_vecs = np.random.normal(size=(n_vecs, d))
    base_vecs /= np.sqrt(np.sum(base_vecs ** 2, axis=1)).reshape(n_vecs, 1)

    return base_vecs


def sents_to_embeddings(embeddings, d_emb, sents):
    sent_vecs = np.zeros(shape=(len(sents), d_emb))
    for idx, sent in enumerate(sents):
        for token in sent:
            sent_vecs[idx] += np.array(embeddings.emb(token.lower(), "zero"))
        sent_vecs[idx] /= len(sent)

    return sent_vecs


def fast_sents_to_embeddings(embeddings, d_emb, sents):
    word_to_sents_map = defaultdict(list)
    for idx, sent in enumerate(sents):
        for tok in sent:
            word_to_sents_map[tok.lower()].append(idx)

    sent_vecs = np.zeros(shape=(len(sents), d_emb))
    for tok, sent_indices in word_to_sents_map.items():
        sent_vecs[sent_indices,:] += np.array(embeddings.emb(tok, "zero"))

    for idx, sent in enumerate(sents):
        sent_vecs[idx] /= len(sent)

    return sent_vecs


def cluster_partioning_glove(sorted_sent_ids, id_sentence_map, num_clusters):
    d_emb = 300
    embeddings = GloveEmbedding('common_crawl_840', d_emb=d_emb, show_progress=True)
    sents = []
    for sent_id in sorted_sent_ids:
        sents.append([t[0] for t in id_sentence_map[sent_id]])
    sent_vecs = sents_to_embeddings(embeddings, d_emb, sents)

    n_vecs = 18
    base_vecs = random_unit_vecs(d_emb, n_vecs)

    sims = cosine_similarity(sent_vecs, base_vecs)

    sims = sims >= 0

    partition_ids = {}
    sent_partitions = {}
    curr_partition_id = 0

    for idx, sim_vec in enumerate(sims):
        part_id = partition_ids.get(tuple(sim_vec))
        if part_id is None:
            part_id = curr_partition_id
            curr_partition_id += 1
            partition_ids[tuple(sim_vec)] = part_id
        sent_partitions[idx] = part_id

    return sent_partitions


class ConstantScoreFactor:
    def __init__(self, id_reward_map):
        self.id_reward_map = id_reward_map

    def find_score_delta(self, sent):
        return self.id_reward_map[sent]

    def update_scores(self, new_sentence):
        pass


class RedundancyFactor:
    @classmethod
    def from_sentences(cls, rewards, id_sentence_map, cluster_id_map, num_clusters=None, clusterer=cluster_kmeans_glove):
        #model = TfidfVectorizer(stop_words="english")
        sorted_sent_ids = sorted(id_sentence_map.keys())

        #tf_idf = model.fit_transform(map(lambda sid: " ".join(
        #    map(lambda t: t[0], id_sentence_map[sid])), sorted_sent_ids))
        # d_emb = 300
        # embeddings = GloveEmbedding('common_crawl_840', d_emb=d_emb, show_progress=True)
        # candidate_vecs = fast_sents_to_embeddings(embeddings, d_emb, [[t[0] for t in id_sentence_map[sid]] for sid in sorted_sent_ids])
        # from sklearn.manifold import TSNE
        # from sklearn.decomposition import PCA
        # import MulticoreTSNE


        #clustering = [0] * len(sorted_sent_ids)

        clustering = cluster_same_words(sorted_sent_ids, id_sentence_map, num_clusters, cluster_id_map=cluster_id_map)
        sent_partitions = {}
        clusters = defaultdict(list)

        for cl_id, sent_id in zip(clustering, sorted_sent_ids):
        #    print(sent_id, cl_id)
            sent_partitions[sent_id] = cl_id
            clusters[cl_id].append(sent_id)

        # print("Run TSNE")
        # X_embedded = MulticoreTSNE.MulticoreTSNE(n_components=2, n_jobs=6).fit_transform(PCA(n_components=50).fit_transform(candidate_vecs))
# 
       # X_embedded = PCA(n_components=2).fit_transform(candidate_vecs)
        # import matplotlib.pyplot as plt
        # print("Plot")
        #iterator = map(lambda x: "".join(x), it.product(iter("rgbycmk"), iter("o^.,v<>138")))
        # colors = it.cycle(iterator)
        #for cluster_sent_ids, color in zip(clusters.values(), colors):
        #    plt.plot(X_embedded[cluster_sent_ids,0], X_embedded[cluster_sent_ids,1], color)
        # plt.show()
# 
        return RedundancyFactor(rewards, sent_partitions)

    def __init__(self, rewards, sent_partitions, normalize=True):
        self.rewards = rewards
        self.sent_partitions = sent_partitions

        self.redundant_sentences = set()

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

        if self.reward_sums[new_sent_partition_id] != 0:
            self.redundant_sentences.add(new_sentence)

        self.reward_sums[new_sent_partition_id] += self.rewards[new_sentence]
        self.reward_sqrts[new_sent_partition_id] = math.sqrt(self.reward_sums[new_sent_partition_id])

    def find_score_delta(self, sent):
        sent_partition_id = self.sent_partitions[sent]
        delta_s = math.sqrt(self.reward_sums[sent_partition_id] + self.rewards[sent]) - self.reward_sqrts[sent_partition_id]

        return delta_s


def precompute_hash_sims(cand_hashes, buckets):
        precomputed_hash_sims = {}
        for cand_hash in cand_hashes:
            if precomputed_hash_sims.get(cand_hash) is not None:
                continue

            precomputed_hash_sims[cand_hash] = {}

            for doc_hash in buckets:
                matches = [c == d for c, d in zip(cand_hash, doc_hash)]
                precomputed_hash_sims[cand_hash][doc_hash] = sum(matches) / len(matches)
        return precomputed_hash_sims


def fast_precompute_hash_sims(cand_hashes, buckets):
    hashlen = len(next(iter(buckets)))

    hashes_with_1_at_position = [[] for _ in range(hashlen)]

    for doc_hash in buckets:
        for idx, v in enumerate(doc_hash):
            if v:
                hashes_with_1_at_position[idx].append(doc_hash)

    precomputed_hash_sims = {}
    for cand_hash in cand_hashes:
        if precomputed_hash_sims.get(cand_hash) is not None:
            continue

        hashsim_dict = defaultdict(lambda: 0.0)
        precomputed_hash_sims[cand_hash] = hashsim_dict

        for idx, v in enumerate(cand_hash):
            if not v:
                continue

            for dhash in hashes_with_1_at_position[idx]:
                hashsim_dict[dhash] += 1.0 / hashlen

    print(list(hashsim_dict.values()))

    #print(precomputed_hash_sims)
    return precomputed_hash_sims


class BucketedRedundancyFactor:
    @classmethod
    def from_sentences(cls, rewards, doc_sents, id_sentence_map, normalize=False):
        #model = TfidfVectorizer(stop_words="english")
        start_time = time.time()

        sorted_sent_ids = sorted(id_sentence_map.keys())

        d_emb = 300
        embeddings = GloveEmbedding('common_crawl_840', d_emb=d_emb, show_progress=True)
        sent_vecs = fast_sents_to_embeddings(embeddings, d_emb, [s.as_token_attr_sequence("form_lowercase") for s in doc_sents])

        n_vecs = 16
        base_vecs = random_unit_vecs(d_emb, n_vecs)

        doc_sims = cosine_similarity(sent_vecs, base_vecs)
        doc_hashes = doc_sims >= 0

        logger.debug("Computed doc sentences hashes (time: {}s)".format(time.time() - start_time))
        start_time = time.time()

        buckets = Counter(map(tuple, doc_hashes))

        candidate_vecs = fast_sents_to_embeddings(embeddings, d_emb, [[t[0] for t in id_sentence_map[sid]] for sid in sorted_sent_ids])

        cand_sims = cosine_similarity(candidate_vecs, base_vecs)

        cand_hashes = [tuple(h) for h in cand_sims >= 0]

        logger.debug("Computed candidate hashes (time: {}s)".format(time.time() - start_time))
        start_time = time.time()

        precomputed_hash_sims = fast_precompute_hash_sims(cand_hashes, buckets)

        overlaps = {}
        per_cand_hashes = {}

        for sent_id, hash_ in zip(sorted_sent_ids, cand_hashes):
            #overlaps[sent_id] = precomputed_hash_sims[hash_]
            per_cand_hashes[sent_id] = hash_

        return BucketedRedundancyFactor(rewards, buckets, per_cand_hashes, precomputed_hash_sims, normalize=normalize)

    def __init__(self, rewards, buckets, cand_hashes, cand_hash_overlaps, normalize=True):
        self.rewards = rewards
        self.buckets = buckets
        self.cand_hashes = cand_hashes
        self.cand_hash_overlaps = cand_hash_overlaps

        if normalize:
            self._normalize_rewards()

        self.bucket_coverages = dict((b, 0.0) for b in self.buckets)

        self.newness_scores = dict((h, 1.0) for h in self.cand_hash_overlaps)

    def _normalize_rewards(self):
        max_reward = 0
        for reward in self.rewards.values():
            max_reward = max(reward, max_reward)

        for sid, reward in list(self.rewards.items()):
            self.rewards[sid] = reward / max_reward

    def update_scores(self, new_sentence):
        for bucket, overlap in self.cand_hash_overlaps[self.cand_hashes[new_sentence]].items():
            self.bucket_coverages[bucket] = max(overlap, self.bucket_coverages[bucket])

            print(max(overlap, self.bucket_coverages[bucket]))

        for cand_hash, overlaps in self.cand_hash_overlaps.items():
            bucket_size_sum = 0.0
            total_newness_score = 0.0
            for bucket, overlap in overlaps.items():
                bucket_size = self.buckets[bucket]
                bucket_overlap = min(overlap, 1 - self.bucket_coverages[bucket])

                bucket_size_sum += bucket_size

                total_newness_score += bucket_overlap * bucket_size

            self.newness_scores[cand_hash] = total_newness_score / bucket_size_sum

    def find_score_delta(self, sent):
        factor = self.newness_scores[self.cand_hashes[sent]]

        #print(factor)

        return factor * self.rewards[sent]


class NGramCoverageFactor:
    def __init__(self, id_sentence_map, word_score_map):
        self.covered_words = set()
        self.id_sentence_map = id_sentence_map
        self.word_score_map = {}

        max_val = max(word_score_map.values())

        for key, val in word_score_map.items():
            self.word_score_map[key] = word_score_map[key] / max_val

    def find_score_delta(self, sent_id):
        sent = self.id_sentence_map[sent_id]
        all_words = set(sent)
        new_words = all_words - self.covered_words

        left_over_score = 0
        for word in new_words:
            left_over_score += self.word_score_map.get(word, 0)

        return left_over_score / len(all_words)

    def update_scores(self, new_sent_id):
        new_sentence = self.id_sentence_map[new_sent_id]
        self.covered_words.update(new_sentence)


def create_compressed_sent_repr(sent_vecs, num_base_vecs):
            base_vecs = random_unit_vecs(sent_vecs.shape[1], num_base_vecs)

            sims = cosine_similarity(sent_vecs, base_vecs)

            directions = sims >= 0.0

            num_matches = np.sum(directions, axis=0)

            return base_vecs, num_matches



class BucketedCoverageFactor:
    @classmethod
    def from_sentences(cls, doc_sents, id_sentence_map, id_date_map, num_date_anchors=100, num_base_vecs=100, normalize=False):
        start_time = time.time()

        date_freqs = Counter(s.predicted_date for s in doc_sents)

        sorted_dates_with_freq = sorted(date_freqs.items())
        num_sents = len(doc_sents)

        sents_per_bucket = num_sents // num_date_anchors

        date_buckets = {}
        curr_bucket = 0
        curr_freq_sum = 0
        for date, freq in sorted_dates_with_freq:
            date_buckets[date] = curr_bucket
            curr_freq_sum += freq

            if curr_freq_sum >= sents_per_bucket and curr_bucket < num_date_anchors:
                curr_bucket += 1
                curr_freq_sum = 0

        sent_and_date_buckets = defaultdict(lambda: ([], set()))

        for sent in doc_sents:
            date = sent.predicted_date
            bucket_sents, bucket_dates = sent_and_date_buckets[date_buckets[date]]

            bucket_sents.append(sent)
            bucket_dates.add(date)

        logger.debug("Computed date buckets (time: {}s)".format(time.time() - start_time))
        start_time = time.time()

        d_emb = 300
        embeddings = GloveEmbedding('common_crawl_840', d_emb=d_emb, show_progress=True)

        checkpoints = []
        for sents, dates in sent_and_date_buckets.values():
            sent_vecs = sents_to_embeddings(embeddings, d_emb, [s.as_token_attr_sequence("form_lowercase") for s in sents])
            base_vecs, num_matches = create_compressed_sent_repr(sent_vecs, num_base_vecs)

            center_date = min(dates) + (max(dates) - min(dates)) / 2

            checkpoints.append((center_date, base_vecs, num_matches, len(sents)))

        logger.debug("Computed checkpoints (time: {}s)".format(time.time() - start_time))
        start_time = time.time()

        sents_by_date = defaultdict(list)
        for sent_id, sent in id_sentence_map.items():
            sent_date = id_date_map[sent_id]
            sents_by_date[sent_date].append(sent_id)

        sent_scores = {}
        for idx, (sents_date, sent_ids) in enumerate(sents_by_date.items()):
            print("{}/{}".format(idx, len(sents_by_date)))
            sent_sims = np.zeros(len(sent_ids))
            factor_sum = 0.0
            sent_vecs = fast_sents_to_embeddings(embeddings, d_emb, [[t[0] for t in id_sentence_map[id_]] for id_ in sent_ids])
            for check_date, base_vecs, num_matches, num_members in checkpoints:
                factor = 1.0 / (abs((check_date - sents_date).days) + 1)

                sent_signatures = (cosine_similarity(sent_vecs, base_vecs) >= 0.0).astype(np.float32)
                sent_signatures *= num_matches.reshape(1, len(base_vecs))
                sent_signatures /= num_members

                cosine_sims = np.average(sent_signatures, axis=1)
                sent_sims += cosine_sims * factor

                factor_sum += factor
            sent_sims /= factor_sum

            for sid, score in zip(sent_ids, sent_sims):
                sent_scores[sid] = score

        logger.debug("Computed scores (time: {}s)".format(time.time() - start_time))

        return BucketedCoverageFactor(sent_scores, normalize=normalize)

    def __init__(self, sent_scores, normalize=True):
        self.sent_scores = sent_scores
        if normalize:
            self._normalize_rewards()

    def update_scores(self, new_sentence):
        pass

    def find_score_delta(self, sent):
        delta_s = self.sent_scores[sent]
        return delta_s

    def _normalize_rewards(self):
        max_reward = 0
        for sent_score in self.sent_scores.values():
            max_reward = max(sent_score, max_reward)

        for sid, reward in list(self.sent_scores.items()):
            self.sent_scores[sid] = reward / max_reward


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

        self.buckets = defaultdict(list)

        for sent, length in self.sent_sizes.items():
            self.buckets[length].append(sent)

    def check(self, sent):
        new_size = self.current_size + self.sent_sizes[sent]

        if new_size > self.knapsack_size:
            return False
        else:
            return True

    def update(self, new_sent, filtered_sentences):
        self.current_size += self.sent_sizes[new_sent]

        return [sent for length, sents in self.buckets.items() for sent in sents if self.current_size + length > self.knapsack_size]


class SimilarityConstraint:
    def __init__(self, id_sentence_map, threshold=0.5):
        self.id_sentence_map = id_sentence_map

        sorted_sent_ids = sorted(id_sentence_map.keys())

        d_emb = 300
        embeddings = GloveEmbedding('common_crawl_840', d_emb=d_emb, show_progress=True)

        model = TfidfVectorizer(stop_words="english", max_features=1024, min_df=5)
        tf_idf = model.fit_transform([" ".join([t[0] for t in id_sentence_map[sid]]) for sid in sorted_sent_ids])

        #sent_vecs = fast_sents_to_embeddings(embeddings, d_emb, )

        self.tf_idf = tf_idf

        #self.id_vec_map = dict((sorted_sent_ids[idx], vec) for idx, vec in enumerate(sent_vecs))

        self.threshold = threshold

    def update(self, new_sent, filtered_sentences):
        to_remove = set()

        relevant_sids = np.array(list(set(self.id_sentence_map.keys()) - filtered_sentences))
        sims = cosine_similarity(self.tf_idf[new_sent], self.tf_idf[list(relevant_sids)])

        to_remove = relevant_sids[(sims > 0.5).nonzero()[1]]

        print(len(to_remove))

        return to_remove


class SubsetKnapsackConstraint:
    def __init__(self, knapsack_size, sent_sizes, relevant_sents):
        self.knapsack_size = knapsack_size
        self.sent_sizes = sent_sizes
        self.current_size = 0
        self.relevant_sents = set(relevant_sents)

        self.buckets = defaultdict(list)

        for sent, length in self.sent_sizes.items():
            if sent in self.relevant_sents:
                self.buckets[length].append(sent)

    def check(self, sent):
        if sent not in self.relevant_sents:
            return True
        new_size = self.current_size + self.sent_sizes[sent]

        if new_size > self.knapsack_size:
            return False
        else:
            return True

    def update(self, new_sent, filtered_sentences):
        if new_sent in self.relevant_sents:
            self.current_size += self.sent_sizes[new_sent]

            removed_candidates = [sent for length, sents in self.buckets.items() for sent in sents if self.current_size + length > self.knapsack_size]

            return removed_candidates

        return []


class ConstantSizeSubsetKnapsackConstraint:
    def __init__(self, knapsack_size, relevant_sents):
        self.knapsack_size = knapsack_size
        self.current_size = 0
        self.relevant_sents = set(relevant_sents)

    def update(self, new_sent, filtered_sentences):
        if new_sent in self.relevant_sents:
            self.current_size += 1

            if self.current_size >= self.knapsack_size:
                return self.relevant_sents

        return []


class ClusterMembershipConstraint:
    def __init__(self, id_cluster_map):
        self.id_cluster_map = id_cluster_map
        self.included_clusters = set()

        self.cluster_sentences = defaultdict(list)

        for sent, cluster in self.id_cluster_map.items():
            self.cluster_sentences[cluster].append(sent)

    def check(self, sent):
        return self.id_cluster_map[sent] not in self.included_clusters

    def update(self, new_sent, filtered_sentences):
        self.included_clusters.add(self.id_cluster_map[new_sent])
        return self.cluster_sentences[self.id_cluster_map[new_sent]]

from collections import defaultdict


class MaxDateCountConstraint:
    def __init__(self, max_date_count, sent_dates):
        self.selected_dates = set()
        self.sent_dates = sent_dates
        self.max_date_count = max_date_count

        self.date_sents = defaultdict(list)

        self.has_reached_limit = False

        for sent, date in self.sent_dates.items():
            self.date_sents[date].append(sent)

    def check(self, sent):
        date = self.sent_dates[sent]
        if date not in self.selected_dates and len(self.selected_dates) >= self.max_date_count:
            return False

        return True

    def update(self, new_sent, filtered_sentences):
        date = self.sent_dates[new_sent]
        self.selected_dates.add(date)

        if len(self.selected_dates) < self.max_date_count:
            return []
        elif not self.has_reached_limit:
            self.has_reached_limit = True
            return [sent for date, sents in self.date_sents.items() for sent in sents if date not in self.selected_dates]
        else:
            return []


class SubModularOptimizer:
    def __init__(self, factors, constraints, selection_callback=None):
        self.factors = factors
        print(len(self.factors))
        self.constraints = constraints

        self.selection_callback = selection_callback

    def run(self, sentences):
        selected_sentences = set()
        filtered_sentences = set()

        #was_filtered = [False for _ in sentences]

        while True:
            best_sentence = None
            best_delta_s = 0
            for idx, sent in enumerate(sentences):
                if sent in filtered_sentences or sent in selected_sentences:
                    continue
                #fullfills_constr = self._does_pass_constraints(sent)
#
                #if not fullfills_constr:
                #    was_filtered[idx] = True
                #    continue

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
                    new_excluded_sentences = constraint.update(best_sentence, filtered_sentences)
                    filtered_sentences.update(new_excluded_sentences)

        logger.info("Sentence selection complete")
        return selected_sentences

    def _determine_delta_s(self, sent):
        delta_s_sum = 0.0
        for factor in self.factors:
            delta_s_sum += factor.find_score_delta(sent)

        return delta_s_sum

    def _does_pass_constraints(self, sent):
        return all(constr.check(sent) for constr in self.constraints)
