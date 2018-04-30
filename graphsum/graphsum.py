from scipy.sparse import lil_matrix, csr_matrix
import scipy.sparse.linalg as sp_linalg
from networkx import DiGraph
import networkx
import matplotlib.pyplot as plt

from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.regexp import WhitespaceTokenizer
import nltk
from pymprog import model

from collections import Counter, defaultdict, deque

import os
import sys
import numpy as np
import networkx as nx
import itertools as it
import random
import math
from networkx.drawing.nx_pydot import to_pydot
import functools
from utils import iter_files, scnd, fst, is_punctuation
from langmodel import StoredLanguageModel, KenLMLanguageModel
from reader import StanfordXMLReader, Sentence, Token
from similarity import EmbeddingsCosingSimilarityModel, read_glove_embeddings, ModifiedIdfCosineSimilarityModel, SklearnTfIdfCosineSimilarityModel, BinaryCosineSimilarityModel, BinaryOverlapSimilarityModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import time

import logging


class ClusterGenerator:
    def __init__(self, clustering_func, min_cluster_size, cache_dir="clusters"):
        self.clustering_func = clustering_func
        self.min_cluster_size = min_cluster_size
        self.cache_dir = cache_dir

    def cluster_from_documents(self, documents, cache_key="", clustering_input=None):
        if clustering_input is None:
            clustering_input = documents
        func_name = self.clustering_func.__name__

        cache_path = os.path.join(self.cache_dir, func_name + ":" + cache_key)

        if os.path.isdir(cache_path):
            cluster_dict = read_clusters(cache_path, documents)
        else:
            os.makedirs(cache_path)
            cluster_dict = self.clustering_func(clustering_input)
            self._save_clusters(cluster_dict, cache_path)

        for cluster_id, members in list(cluster_dict.items()):
            if len(members) < self.min_cluster_size:
                del cluster_dict[cluster_id]

        return cluster_dict

    def _save_clusters(self, cluster_dict, cache_path):
        for cl_id, members in cluster_dict.items():
            fname = os.path.join(cache_path, "cl-{}.txt".format(cl_id))
            save_cluster(fname, members)


def order_clusters_mayority(cluster_dict):
    """
    Returns list of ids in appropriate order
    """
    def cluster_mayority_cmp(cl1, cl2):
        c_1_mayority = 0
        id1, members_1 = cl1
        id2, members_2 = cl1

        for sent_1 in members_1:
            for sent_2 in members_2:
                if sent_1.document == sent_2.document:
                    if sent_1.idx > sent_2.idx:
                        c_1_mayority += 1
                    elif sent_1.idx < sent_2.idx:
                        c_1_mayority -= 1

        return c_1_mayority

    return list(map(fst, sorted(cluster_dict.items(), key=functools.cmp_to_key(cluster_mayority_cmp))))


class SummarizationPipeline:
    def __init__(self, clustering_func, lm, global_tr=True):
        self.clustering_func = clustering_func
        self.lm = lm

        self.global_tr = global_tr

    def summarize_documents(self, documents, cache_key=""):
        similarity_model = BinaryCosineSimilarityModel(STOPWORDS)
        similarity_model.fit([sent.as_token_attr_sequence("form") for doc in documents for sent in doc])

        cluster_gen = ClusterGenerator(cluster_without_seed, 5)
        clusters = cluster_gen.cluster_from_documents(documents, cache_key=cache_key)

        if len(clusters) == 0:
            return ""

        sorted_clusters = order_clusters_mayority(clusters)
        cluster_indices = dict(map(lambda x: (x[1], x[0]), enumerate(sorted_clusters)))

        per_cluster_candidates = []
        print("Found {} clusters".format(len(clusters)))
        print("Generating cluster candidates...")
        if self.global_tr:
            tr_scores = calculate_keyword_text_rank([sent.as_token_tuple_sequence("form_lowercase", "pos") for doc in documents for sent in doc])
        else:
            tr_scores = None

        cluster_idx_id_map = {}
        for cluster_idx, (cluster_id, cluster_sentences) in enumerate(clusters.items()):
            candidates = generate_summary_candidates(
                list(
                    map(lambda s: s.as_token_tuple_sequence("form_lowercase", "pos"),
                        cluster_sentences)), self.lm, tr_scores=tr_scores)
            per_cluster_candidates.append(candidates)
            cluster_idx_id_map[cluster_idx] = cluster_id
            print("Found {} candidates for cluster".format(len(candidates)))
        print("Selecting sentences...")

        flat_sentences = [(sent, score) for sents in per_cluster_candidates for (sent, score) in sents]
        print("Best sentences overall")
        print(flat_sentences[0])
        for sent in sorted(flat_sentences, key=lambda x: x[1], reverse=True)[:10]:
            print(
                " ".join(map(lambda t: t[0], sent[0])),
                "Score:", sent[1],
                "Gram:", 1.0 / (1.0 - self.lm.estimate_sent_log_proba(list(map(lambda x: x[0], sent[0])))),
                "Info:", calculate_informativeness(sent[0], tr_scores))
        print("-----")

        print("Best sentences informativeness")
        for sent in sorted(flat_sentences, key=lambda s: calculate_informativeness(s[0], tr_scores), reverse=True)[:10]:
            print(
                " ".join(map(lambda t: t[0], sent[0])),
                "Score:", sent[1],
                "Gram:", 1.0 / (1.0 - self.lm.estimate_sent_log_proba(list(map(lambda x: x[0], sent[0])))),
                "Info:", calculate_informativeness(sent[0], tr_scores))
        print("-----")

        #sentences = select_sentences(per_cluster_candidates, similarity_model)
        sentences = select_sentences_submod(
            per_cluster_candidates,
            [sent.as_token_tuple_sequence("form_lowercase", "pos") for doc in documents for sent in doc]
        )

        print("Ordering...")
        sentences = sorted(sentences, key=lambda x: cluster_indices.get(cluster_idx_id_map[x[1]], 0))
        plaintext_sents = list(map(lambda x: x[0], sentences))

        return "\n".join(plaintext_sents)


from submodular import SubModularOptimizer, RedundancyFactor, CoverageFactor, KnapsackConstraint



def select_sentences_submod(per_cluster_candidates, doc_sents, max_tokens=None, max_sents=2):
    id_sentence_map = {}
    id_cluster_map = {}
    id_score_map = {}
    id_tok_count_map = {}

    sent_idx_counter = 0
    for cl_idx, members in enumerate(per_cluster_candidates):
        for sent, score in members:
            id_sentence_map[sent_idx_counter] = sent
            id_cluster_map[sent_idx_counter] = cl_idx
            id_score_map[sent_idx_counter] = score
            id_tok_count_map[sent_idx_counter] = len(sent)

            sent_idx_counter += 1

    constraints = []

    if max_sents is not None:
        constraints.append(KnapsackConstraint(max_sents, defaultdict(lambda: 1)))

    if max_tokens is not None:
        constraints.append(KnapsackConstraint(max_tokens, id_tok_count_map))

    #cluster_redundancy_factor = RedundancyFactor(id_score_map, id_cluster_map)

    kmeans_redundancy_factor = RedundancyFactor.from_sentences(
        id_score_map,
        id_sentence_map)

    coverage_factor = CoverageFactor.from_sentences(
        doc_sents,
        list(map(scnd, sorted(id_sentence_map.items(), key=fst)))
    )

    opt = SubModularOptimizer(
        [
            kmeans_redundancy_factor,
            coverage_factor
        ],
        constraints)

    sent_ids = opt.run(range(sent_idx_counter))

    selected_sentences = []
    for sent_id in sent_ids:
        selected_sentences.append((" ".join([tok for tok, pos in id_sentence_map[sent_id]]), id_cluster_map[sent_id]))

    return selected_sentences


class SentenceCompressionGraph:
    def __init__(self, stopwords):
        self.stopwords = set(stopwords)
        self.graph = DiGraph()
        self.graph.add_node("START")
        self.graph.add_node("END")

        self.surface_to_nodes_map = defaultdict(set)

    def add_sentences(self, sentences):
        for sent in sentences:
            self.add_sentence(sent)

    def add_sentence(self, sentence):
        normalized_sentence = []
        for tok in sentence:
            assert len(tok) == 2, "Expecting two-tuple of (form, pos)"
            form, pos = tok
            normalized_sentence.append((form.lower(), pos))

        token_to_node_map = self.map_tokens_to_nodes(normalized_sentence)

        prev_node = "START"

        for t_idx, token in enumerate(normalized_sentence):
            node = token_to_node_map[t_idx]
            if node is None:
                node = self.create_new_node(token)

            self.surface_to_nodes_map[token].add(node)

            edge = self.graph[prev_node].get(node)

            if edge is None:
                self.graph.add_edge(prev_node, node, frequency=1)
            else:
                edge["frequency"] += 1

            self.graph.nodes[node].setdefault("mapped_tokens", []).append((t_idx, token, sentence))
            prev_node = node

        end_edge = self.graph[prev_node].get("END")
        if end_edge is None:
            self.graph.add_edge(prev_node, "END", frequency=1)
        else:
            end_edge["frequency"] += 1

    def create_new_node(self, token):
        node_id = len(self.graph.nodes)
        label = "/".join(token)
        label = label.replace(",", "\\,")
        self.graph.add_node(node_id, token=token, label=label)

        return node_id

    def map_tokens_to_nodes(self, sentence):
        non_stopword_tokens = set()
        stopword_tokens = set()
        punctuation_tokens = set()

        for t_idx, (form, pos) in enumerate(sentence):
            if form in self.stopwords:
                stopword_tokens.add(t_idx)
            elif is_punctuation(form):
                punctuation_tokens.add(t_idx)
            else:
                non_stopword_tokens.add(t_idx)

        mappings = {}
        non_stopword_mappings = self._map_tokens_to_nodes(non_stopword_tokens, sentence, set())
        # CHECKME: This should be unneccesary
        disallowed_nodes = set(non_stopword_mappings.values())
        stopword_mappings = self._map_tokens_to_nodes(stopword_tokens, sentence, disallowed_nodes, True)
        disallowed_nodes.update(set(stopword_mappings.values()))
        punct_mappings = self._map_tokens_to_nodes(punctuation_tokens, sentence, disallowed_nodes, True)

        mappings.update(non_stopword_mappings)
        mappings.update(stopword_mappings)
        mappings.update(punct_mappings)

        return mappings

    def _map_tokens_to_nodes(self, tokens, sentence, disallowed_nodes, require_overlap=False):
        token_mappings = {}
        node_token_mapping = {}
        token_candidate_preferences = {}

        for t_idx, token in enumerate(sentence):
            if t_idx not in tokens:
                continue

            form, pos = token

            #if form in self.stopwords or not form.isalnum():
            #    continue

            candidate_mappings = self.surface_to_nodes_map.get(token, [])
            candidate_mappings = list(filter(lambda n: n not in disallowed_nodes, candidate_mappings))

            if require_overlap:
                candidate_mappings = list(filter(lambda n: self.compute_token_overlap(sentence, t_idx, n) > 0, candidate_mappings))

            if len(candidate_mappings) == 0:
                canidate_preferences = []
            elif len(candidate_mappings) == 1:
                node = candidate_mappings[0]
                canidate_preferences = [node]
            else:
                keyfunc = lambda c_node: (self.compute_token_overlap(sentence, t_idx, c_node), len(self.graph.nodes[c_node]["mapped_tokens"]))
                canidate_preferences = sorted(candidate_mappings, key=keyfunc, reverse=True)

            token_candidate_preferences[t_idx] = canidate_preferences

        unmapped_tokens_indices = set(tokens)

        while len(unmapped_tokens_indices) > 0:
            t_idx = unmapped_tokens_indices.pop()
            preferences = token_candidate_preferences[t_idx]

            while len(preferences) > 0:
                best_node = preferences.pop()

                prev_t_idx = node_token_mapping.get(best_node)

                if prev_t_idx is None:
                    token_mappings[t_idx] = best_node
                    node_token_mapping[best_node] = t_idx
                    break
                else:
                    prev_t_overlap = self.compute_token_overlap(sentence, prev_t_idx, best_node)
                    new_t_overlap = self.compute_token_overlap(sentence, t_idx, best_node)

                    if new_t_overlap > prev_t_overlap:
                        token_mappings[t_idx] = best_node
                        node_token_mapping[best_node] = t_idx
                        unmapped_tokens_indices.add(prev_t_idx)
                        break
            else:
                token_mappings[t_idx] = None

        return token_mappings

    def compute_token_overlap(self, sentence, t_idx, node, filter_stopwords=True):
        left_ctx_tokens = self.get_node_context_tokens(node, "l")
        right_ctx_tokens = self.get_node_context_tokens(node, "r")

        overlap = 0

        if t_idx < len(sentence) - 1:
            r_ctx = sentence[t_idx + 1]
            if (not filter_stopwords or r_ctx[0] not in self.stopwords) and r_ctx in right_ctx_tokens:
                overlap += 1
        if t_idx > 0:
            l_ctx = sentence[t_idx - 1]
            if (not filter_stopwords or l_ctx[0] not in self.stopwords) and l_ctx in left_ctx_tokens:
                overlap += 1

        return overlap

    def get_node_context_tokens(self, node, dir_):
        if dir_ == "l":
            neighbours = self.graph.predecessors(node)
        elif dir_ == "r":
            neighbours = self.graph.successors(node)
        else:
            raise RuntimeError("dir must be either 'l' or 'r'")

        tokens = []

        for n in neighbours:
            tok = self.graph.nodes[n].get("token")
            if tok is not None:
                tokens.append(tok)

        return tokens

    def calculate_strong_links_weights(self):
        max_freq = 0

        for src, trg, data in self.graph.edges(data=True):
            if src == "START" or trg == "END":
                data["weight_sl"] = 1.0 / data["frequency"]  # TODO: Check what this should be
                data["label"] = data["weight_sl"]
                continue

            src_freq = len(self.graph.nodes[src]["mapped_tokens"])
            trg_freq = len(self.graph.nodes[trg]["mapped_tokens"])

            diff = 0.0

            for t_idx_1, _, sent_1 in self.graph.nodes[src]["mapped_tokens"]:
                for t_idx_2, _, sent_2 in self.graph.nodes[trg]["mapped_tokens"]:
                    if sent_1 == sent_2 and t_idx_1 < t_idx_2:
                        diff += (t_idx_2 - t_idx_1) ** -1

            max_freq = max(max_freq, data["frequency"])

            w = (src_freq + trg_freq) / diff  # data["frequency"]

            w /= src_freq * trg_freq

            data["weight_sl"] = w
            data["label"] = str(w)

        for src, trg, data in self.graph.edges(data=True):
            data["rel_frequency"] = data["frequency"] / max_freq

    def generate_compression_candidates(self, n=2500, minlen=8, maxlen=None, filterfunc=lambda c: True, use_weighting=True, timeout=600, return_weight=False, max_tries=2500):
        self.calculate_strong_links_weights()

        num_yielded = 0
        candidate_set = set()

        start_time = time.time()

        logging.info("Processing graph V = {}, E = {}".format(len(self.graph), len(self.graph.edges)))

        if use_weighting:
            import itertools
            paths = list(itertools.islice(nx.all_simple_paths(self.graph, "START", "END"), n))

            if len(paths) >= n:
                path_iter = nx.shortest_simple_paths(self.graph, "START", "END", weight="weight_sl")
            else:
                path_iter = iter(paths)
        else:
            path_iter = nx.all_simple_paths(self.graph, "START", "END")

        num_tries = 0

        for path in path_iter:
            if max_tries is not None and num_tries > max_tries:
                break
            num_tries += 1

            if time.time() - start_time >= timeout:
                logging.warn("Timeout during sentence generation")
                break
            if len(path) < minlen + 2:  # Account for START and END
                continue

            if maxlen is not None and len(path) - 2 > maxlen:   # Account for START and END
                continue

            tokens = [self.graph.nodes[node]["token"] for node in path[1:-1]]

            for form, pos in tokens:
                if pos[0] == "V":
                    break
            else:
                continue

            if not filterfunc(tokens):
                continue

            if tuple(tokens) not in candidate_set:
                if return_weight is True:
                    full_len = 0
                    prev_node = path[0]
                    freq_sum = 0
                    for node in path[1:]:
                        full_len += self.graph[prev_node][node]["weight_sl"]
                        freq_sum += self.graph[prev_node][node]["rel_frequency"]
                        prev_node = node
                    yield tokens, {"weight": full_len, "avg_rel_frequency": freq_sum / len(path)}
                else:
                    yield tokens
                num_yielded += 1
                candidate_set.add(tuple(tokens))

            if num_yielded >= n:
                break

        logging.debug("Needed {} tries to create {} items".format(num_tries, num_yielded))


def insert_into_top_n_list(l, new_item, n):
    if len(l) < n:
        l.append(new_item)

    insert_idx = None
    for idx, item in enumerate(l):
        if new_item > item:
            insert_idx = idx

    if insert_idx is not None:
        l.insert(idx, new_item)
        l.pop()

    return l


def rank_documents_pairwise(docset, sim_model):
    #tf_idf = generate_word_occurence_matrix(list(map(lambda d: list(d.tokens), docset)))

    top_n_docs = []

    for doc_idx, doc in enumerate(docset):
        consine_sum = 0.0
        for cmp_doc_idx, cmp_doc in enumerate(docset):
            if doc_idx == cmp_doc_idx:
                continue
            #doc_vec = tf_idf[doc_idx]
            #cmp_vec = tf_idf[cmp_doc_idx]
#
            #cosine = doc_vec.dot(cmp_vec.transpose()).toarray()[0][0] / (sp_linalg.norm(doc_vec) * sp_linalg.norm(cmp_vec))
#
            #if (sp_linalg.norm(doc_vec) * sp_linalg.norm(cmp_vec)) == 0:
            #    cosine = 0
#
            #consine_sum += cosine
            consine_sum += sim_model.compute_similarity(doc.as_token_attr_sequence("form"), cmp_doc.as_token_attr_sequence("form"))

        avg_cosine = consine_sum / (len(docset) - 1)
        insert_into_top_n_list(top_n_docs, (avg_cosine, doc_idx), 1)
        print(avg_cosine, doc_idx)

    print(top_n_docs)

    return top_n_docs


def rank_documents_docset_sim(docset, sim_model):
    docset_tokens = [token for doc in docset for token in doc.as_token_attr_sequence("form")]

    doc_sims = []

    for doc_idx, doc in enumerate(docset):
        cosine_sim = sim_model.compute_similarity(doc.as_token_attr_sequence("form"), docset_tokens)
        doc_sims.append((cosine_sim, doc_idx))

    print(doc_sims)
    return max(doc_sims)[1]



class Document:
    def __init__(self, sentences):
        self.sentences = sentences

    @property
    def tokens(self):
        return [tok for sent in self.sentences for tok in sent]

    def __iter__(self):
        return iter(self.tokens)


#class Sentence:
#    def __init__(self, tokens):
#        self.tokens = tokens
#
#    @property
#    def tokens(self):
#        return [tok for sent in self.tokens]
#
#
#class Token:
#    def __init__(self, word, pos):
#        pass


def cluster_with_seed_sentences(seeds, documents, sim_model):
    sentences = [s for doc in documents for s in doc.sentences]
    #tf_idf_matrix = generate_word_occurence_matrix(seeds + sentences)

    #seeds_tf_idf = tf_idf_matrix[:len(seeds),:]
    #remaining_tf_idf = tf_idf_matrix[len(seeds):,:]

    clusters = dict((idx, (seed, [])) for idx, seed in enumerate(seeds))

    for sent_idx, sent in enumerate(sentences):
        best_seed_idx = None
        best_seed_similarity = None
        for seed_idx, seed in enumerate(seeds):
#            vec_seed = seeds_tf_idf[seed_idx].toarray()
#            vec_doc = remaining_tf_idf[sent_idx].toarray()
#
#            vec_seed = vec_seed.reshape((vec_seed.shape[1],))
#            vec_doc = vec_doc.reshape((vec_doc.shape[1],))
#
#            norm = (np.linalg.norm(vec_doc) * np.linalg.norm(vec_seed))
#            if norm > 0:
#                cosine = np.dot(vec_seed, vec_doc) / norm
#            else:
#                cosine = 0
#
#            print(norm, cosine, np.dot(vec_seed, vec_doc))
#
#            #cosine = vec_seed.dot(vec_doc.T) / (sp_linalg.norm(vec_doc) * sp_linalg.norm(vec_seed))
            

            cosine = sim_model.compute_similarity(
                map(lambda t: t[0], filter(lambda t: t[1][0] in "NV", sent.as_token_tuple_sequence("form", "pos"))),
                map(lambda t: t[0], filter(lambda t: t[1][0] in "NV", seed.as_token_tuple_sequence("form", "pos"))))
            cosine = sim_model.compute_similarity(sent.as_token_attr_sequence("form"), seed.as_token_attr_sequence("form"))

            if best_seed_similarity is None or best_seed_similarity < cosine:
                best_seed_similarity = cosine
                best_seed_idx = seed_idx
        print(best_seed_similarity)
        #print(clusters[best_seed_idx][0].as_tokenized_string())
        #print(sent.as_tokenized_string())
        #print()
        if best_seed_similarity > 0.5:
            clusters[best_seed_idx][1].append(sent)
        #else:
        #    seeds.append(sent)
        #    clusters[len(seeds) - 1] = (sent, [sent])

    return clusters

import string

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
STOPWORDS.add(",")
STOPWORDS.add(".")
STOPWORDS.add("''")
STOPWORDS.add("``")
STOPWORDS.update(string.punctuation)
STOPWORDS.add("'s")
STOPWORDS.add("n't")
STOPWORDS.add("-LRB-")
STOPWORDS.add("-RRB-")


def compute_neighbourhood_overlap(sent, graph, token_idx, node_idx, window_size=2, include_stopwords=True):
    # It is sufficient to filter stopwords from one of the sets
    #sent_neighbours = set(
    #    filter(lambda t: include_stopwords or t[0].lower() not in STOPWORDS,
    #    sent[max(0, token_idx - window_size):token_idx]
    #    + sent[token_idx + 1:token_idx + 1 + window_size]
    #))
    #graph_neighbours = set(map(lambda n: graph.nodes[n].get("token"), filter(lambda n: graph.nodes[n].get("marker") is None, list(graph.successors(node_idx)) + list(graph.predecessors(node_idx)))))

    right_graph_context = set(map(lambda n: graph.nodes[n].get("token"), filter(lambda n: graph.nodes[n].get("marker") is None, list(graph.successors(node_idx)))))
    left_graph_context = set(map(lambda n: graph.nodes[n].get("token"), filter(lambda n: graph.nodes[n].get("marker") is None, list(graph.predecessors(node_idx)))))

    sent_neighbours = set(
        filter(lambda t: include_stopwords or t[0].lower() not in STOPWORDS,
        sent[max(0, token_idx - window_size):token_idx]
        + sent[token_idx + 1:token_idx + 1 + window_size]
    ))


    left_sent_neighbours = set(filter(lambda t: include_stopwords or t[0].lower() not in STOPWORDS,
        sent[max(0, token_idx - window_size):token_idx]))
    right_sent_neighbours = set(filter(lambda t: include_stopwords or t[0].lower() not in STOPWORDS,
        sent[token_idx + 1:token_idx + 1 + window_size]))

    return len(left_sent_neighbours.intersection(left_graph_context)) + len(right_sent_neighbours.intersection(right_graph_context))


def add_sentence_to_graph(graph, sent):
    token_node_map = {}
    token_matches = {}


    for t_idx, token in enumerate(sent):
        if token[0].lower() in STOPWORDS:
            continue

        matches = []
        for node_idx, data in graph.nodes.items():
            if "token" not in data:
                continue
            node_token = data["token"]

            if node_token == token:
                if node_idx not in token_node_map.values():
                    matches.append(node_idx)

        if len(matches) == 1:
            token_node_map[t_idx] = matches[0]
        elif len(matches) == 0:
            new_node_idx = len(graph.nodes)
            graph.add_node(new_node_idx, token=token)
            token_node_map[t_idx] = new_node_idx
        else:
            token_matches[t_idx] = matches

    for t_idx, potential_matches in token_matches.items():
        best_match_node = None
        best_match_overlap = None
        for potential_node in potential_matches:
            if potential_node in token_node_map.values():
                continue

            overlap = compute_neighbourhood_overlap(sent, graph, t_idx, potential_node)

            if best_match_node is None or best_match_overlap < overlap:
                best_match_node = potential_node
                best_match_overlap = overlap

        token_node_map[t_idx] = best_match_node

    stopword_token_matches = {}
    for t_idx, token in enumerate(sent):
        if token[0].lower() not in STOPWORDS:
            continue

        matches = []
        for node_idx, data in graph.nodes.items():
            if "token" not in data:
                continue
            node_token = data["token"]

            if node_token == token and compute_neighbourhood_overlap(sent, graph, t_idx, node_idx, include_stopwords=False) > 0:
                if node_idx not in token_node_map.values():
                    matches.append(node_idx)

        if len(matches) == 1:
            token_node_map[t_idx] = matches[0]
        elif len(matches) == 0:
            new_node_idx = len(graph.nodes)
            graph.add_node(new_node_idx, token=token)
            token_node_map[t_idx] = new_node_idx
        else:
            stopword_token_matches[t_idx] = matches

    for t_idx, potential_matches in stopword_token_matches.items():
        best_match_node = None
        best_match_overlap = None

        for potential_node in potential_matches:
            if potential_node in token_node_map.values():
                continue

            overlap = compute_neighbourhood_overlap(
                sent,
                graph,
                t_idx,
                potential_node)

            if best_match_node is None or best_match_overlap < overlap:
                best_match_node = potential_node
                best_match_overlap = overlap

        token_node_map[t_idx] = best_match_node

    token_mappings = sorted(token_node_map.items())

    sent_start_node_idx = len(graph.nodes)
    graph.add_node(sent_start_node_idx, marker="START", orig_sent=sent)

    sent_end_node_idx = len(graph.nodes)
    graph.add_node(sent_end_node_idx, marker="END", orig_sent=sent)

    prev_node = sent_start_node_idx

    for t_idx, n_idx in token_mappings:
        if n_idx is None:
            n_idx = len(graph.nodes)
            graph.add_node(n_idx, token=sent[t_idx])

        graph.add_edge(prev_node, n_idx)
        prev_node = n_idx

    graph.add_edge(prev_node, sent_end_node_idx)


    graph.add_edge("master_start", sent_start_node_idx)
    graph.add_edge(sent_end_node_idx, "master_end")




def generate_cluster_graph(cluster_sents):
    sent_iter = iter(cluster_sents)

    graph = DiGraph()
    graph.add_node("master_start", marker="MASTER")
    graph.add_node("master_end", marker="MASTER")

    for sent in sent_iter:
        add_sentence_to_graph(graph, sent)

    return graph


def main():
    #lm = StoredLanguageModel.from_file("lm_giga_20k_nvp_3gram/lm_giga_20k_nvp_3gram.arpa")
    lm = KenLMLanguageModel.from_file("langmodel20k_vp_3.bin")
    #lm = None

    document_basedir = sys.argv[1]

    #all_docs = []
    #all_sents = []
    #all_cluster_sents = []
#
    #from nltk.stem.snowball import SnowballStemmer
#
    #stemmer = SnowballStemmer("english")
#
    #for cluster_id in os.listdir(document_basedir):
    #    cluster_sentences = []
#
    #    cluster_dir = os.path.join(document_basedir, cluster_id)
    #    for doc_fname in os.listdir(cluster_dir):
    #        if not doc_fname.endswith(".out"):
    #            continue
    #        reader = StanfordXMLReader()
    #        document = reader.run(os.path.join(cluster_dir, doc_fname))
    #        all_docs.append(list(map(lambda t: stemmer.stem(t), document.as_token_attr_sequence("form"))))
#
    #        cluster_sentences.extend(list(map(lambda s: list(map(stemmer.stem, s)), document.as_sentence_attr_sequence("form"))))
#
    #    all_sents.extend(cluster_sentences)
    #    all_cluster_sents.append(cluster_sentences)
#
    #vectorizer = TfidfVectorizer("english")
    #vectorizer.fit(map(lambda d: " ".join(d), all_docs))
    #tf_idf = vectorizer.transform(map(lambda d: " ".join(d), all_sents))
#
    #start_idx = 0
#
    #for cluster_sents in all_cluster_sents:
    #    if len(cluster_sents) == 0:
    #        continue
    #    end_idx = start_idx + len(cluster_sents)
#
    #    global_sims = cosine_similarity(tf_idf[start_idx:end_idx])
    #    print("G", len(list(set(map(lambda x: tuple(sorted(x)), filter(lambda x: x[0] != x[1], zip(*np.where(global_sims > 0.5))))))), len(all_docs))
#
    #    local_tf_idf = TfidfVectorizer("english").fit_transform(map(lambda d: " ".join(d), cluster_sents))
    #    local_sims = cosine_similarity(local_tf_idf)
    #    print("L", len(list(set(map(lambda x: tuple(sorted(x)), filter(lambda x: x[0] != x[1], zip(*np.where(local_sims > 0.5))))))), len(all_docs))
#
#
    #    start_idx = end_idx
#
#
    #sim_model = SklearnTfIdfCosineSimilarityModel()
    ##sim_model.fit(all_docs)
#
    #tf_idf = TfidfVectorizer("english").fit_transform(map(lambda d: " ".join(d), all_docs))
    #sims = cosine_similarity(tf_idf)
#
    #print(len(list(set(map(lambda x: tuple(sorted(x)), filter(lambda x: x[0] != x[1], zip(*np.where(sims > 0.5))))))))

    for cluster_id in os.listdir(document_basedir):
    #for cluster_id in ["331"]:
        documents = []
        cluster_dir = os.path.join(document_basedir, cluster_id)

        for doc_fname in os.listdir(cluster_dir):
            if not doc_fname.endswith(".out"):
                continue
            reader = StanfordXMLReader()
            document = reader.run(os.path.join(cluster_dir, doc_fname))
            documents.append(document)
        sim_model = SklearnTfIdfCosineSimilarityModel()
        sim_model.fit([tok.form for s in d for tok in s] for d in documents)
        summarization = summarize_documents(documents, lm, sim_model)

        with open("{}.sum.txt".format(cluster_id), "w") as f_out:
            f_out.write(summarization)


def timeline_main():
    document_basedir = sys.argv[1]

    lm = KenLMLanguageModel.from_file("langmodel20k_vp_3.bin")

    best_date_id = None
    best_num_dates = None

    for date_id in os.listdir(document_basedir):
        date_path = os.path.join(document_basedir, date_id)
        num_dates = len(os.listdir(date_path))

        if best_date_id is None or num_dates > best_num_dates:
            best_date_id = date_id
            best_num_dates = num_dates

    best_date_path = os.path.join(document_basedir, best_date_id)
    summarization = summarize_timeline_dir(best_date_path, lm)

    print("-" * 10)
    print(summarization)
    print("-" * 10)


def summarize_timeline_dir(docdir, lm):
    documents = []
    for fname in os.listdir(docdir):
        try:
            reader = StanfordXMLReader()
            document = reader.run(os.path.join(docdir, fname))
            documents.append(document)
        except:
            pass

    summerizer = SummarizationPipeline(cluster_without_seed, lm)
    summarization = summerizer.summarize_documents(documents, cache_key=os.path.basename(docdir))

    return summarization


def cluster_without_seed(documents):
    sim_model = BinaryOverlapSimilarityModel(STOPWORDS)
    clusters = []

    for doc in documents:
        for sent in doc:
            for seed, members in clusters:
                cos_sim = sim_model.compute_similarity(
                    seed.as_token_attr_sequence("form"),
                    sent.as_token_attr_sequence("form")
                )

                if cos_sim > 0.7:
                    members.append(sent)

            clusters.append((sent, [sent]))

    return dict(enumerate(map(scnd, clusters)))


def summarize_documents(documents, lm, similarity_model=None):
    #embeddings_similarity_model = EmbeddingsCosingSimilarityModel(read_glove_embeddings("glove.6B.50d.txt"))
    #similarity_model = ModifiedIdfCosineSimilarityModel(STOPWORDS)
    #similarity_model = BinaryCosineSimilarityModel(STOPWORDS)
    #similarity_model.fit([doc.as_token_attr_sequence("form") for doc in documents])

    if similarity_model is not None:
        similarity_model = SklearnTfIdfCosineSimilarityModel()
        similarity_model.fit([sent.as_token_attr_sequence("form") for doc in documents for sent in doc])

    best_doc = documents[rank_documents_docset_sim(documents, similarity_model)]

    tr_scores = calculate_keyword_text_rank([sent.as_token_tuple_sequence("form_lowercase", "pos") for doc in documents for sent in doc])

    #similarity_model = BinaryCosineSimilarityModel(STOPWORDS)
    #similarity_model = BinaryOverlapSimilarityModel(STOPWORDS)

    clusters = cluster_with_seed_sentences(best_doc.sentences, documents, similarity_model)

    #clusters = {0: (None, [doc.sentences[0] for doc in documents])}

    for cluster_id, (seed, members) in list(clusters.items()):
        print(cluster_id)
        for member in members:
            print(member.as_tokenized_string(), (1.0 / (1.0 - lm.estimate_sent_log_proba(member.as_token_attr_sequence("form")))))
        #if len(members) < len(documents) / 2:
        if len(members) < 5:
            del clusters[cluster_id]

    sorted_clusters = []

    for cluster_id, (seed, members) in clusters.items():
        sorted_clusters.append(cluster_id)

    def cluster_mayority_cmp(cluster_id_1, cluster_id_2):
        c_1_mayority = 0

        cluster_1 = clusters[cluster_id_1]
        cluster_2 = clusters[cluster_id_2]

        for sent_1 in cluster_1[1]:
            for sent_2 in cluster_2[1]:
                if sent_1.document == sent_2.document:
                    if sent_1.idx > sent_2.idx:
                        c_1_mayority += 1
                    elif sent_1.idx < sent_2.idx:
                        c_1_mayority -= 1

        return c_1_mayority

    sorted_clusters.sort(key=functools.cmp_to_key(cluster_mayority_cmp))

    cluster_indices = dict(map(lambda x: (x[1], x[0]), enumerate(sorted_clusters)))

    per_cluster_candidates = []
    print("Found {} clusters".format(len(clusters)))
    print("Generating cluster candidates...")
    cluster_idx_id_map = {}
    for cluster_idx, (cluster_id, (_, cluster_sentences)) in enumerate(clusters.items()):
        candidates = generate_summary_candidates(list(map(lambda s: s.as_token_tuple_sequence("form_lowercase", "pos"), cluster_sentences)), lm, tr_scores=tr_scores)
        per_cluster_candidates.append(candidates)
        cluster_idx_id_map[cluster_idx] = cluster_id
        print("Found {} candidates for cluster".format(len(candidates)))
    print("Selecting sentences...")
    for cluster in per_cluster_candidates:
        for sent, _ in cluster:
            print(" ".join(map(lambda x: x[0], sent)), (1.0 / (1.0 - lm.estimate_sent_log_proba(list(map(lambda x: x[0], sent))))))
    sentences = select_sentences(per_cluster_candidates, similarity_model)

    print("Ordering...")

    sentences = sorted(sentences, key=lambda x: cluster_indices[cluster_idx_id_map[x[1]]])
    plaintext_sents = list(map(lambda x: x[0], sentences))

    return "\n".join(plaintext_sents)


def calculate_informativeness(sent, tr_scores):
    informativeness_score = 0
    for token in set(sent):
        informativeness_score += tr_scores.get(token, 0)

    return informativeness_score


def generate_summary_candidates(sentences, lm, tr_scores=None, length_normalized=False, use_weighting=True, include_path_weight=True):
    compressor = SentenceCompressionGraph(STOPWORDS)
    print("Building Graph...")
    compressor.add_sentences(sentences)

    print("Scoring keywords...")
    if tr_scores is None:
        tr_scores = calculate_keyword_text_rank(sentences)

    #pydot.write_png("out.png")

    print("Extracting candidates...")
    sents_and_scores = []

    cluster_vector_model = TfidfVectorizer()
    cluster_vectors = cluster_vector_model.fit_transform([" ".join(map(lambda x: x[0], sent)) for sent in sentences])

    def check_sent_has_verb(sent):
        return any(map(lambda t: t[1] in {"VB", "VBD", "VBP", "VBZ"}, sent))

    def check_closeness(sent):
        sent_vec = cluster_vector_model.transform([" ".join(map(lambda x: x[0], sent))])
        sims = cosine_similarity(sent_vec, cluster_vectors)

        return all(sims[0,:] < 0.8)

    for proposed_sent in compressor.generate_compression_candidates(filterfunc=check_closeness, use_weighting=use_weighting, maxlen=55, return_weight=include_path_weight):
        if include_path_weight:
            proposed_sent, path_weight = proposed_sent

        plaintext_sent = list(map(lambda x: x[0], proposed_sent))
        lm_score = 1 / (1.0 - lm.estimate_sent_log_proba(plaintext_sent))

        informativeness_score = calculate_informativeness(proposed_sent, tr_scores)

        score = informativeness_score * lm_score
        if include_path_weight:
            score *= path_weight
        if length_normalized:
            score /= len(proposed_sent)

        #score = len(proposed_sent) ** 3
        #print(" ".join(plaintext_sent), score, lm_score, informativeness_score, len(proposed_sent))
        sents_and_scores.append((proposed_sent, score))

    return sents_and_scores


def generate_summary_candidates_preselection(sentences, lm):
    compressor = SentenceCompressionGraph(STOPWORDS)
    print("Building Graph...")
    compressor.add_sentences(sentences)

    print("Scoring keywords...")
    tr_scores = calculate_keyword_text_rank(sentences)

    #pydot.write_png("out.png")

    print("Extracting candidates...")
    sents_and_scores = []

    def check_sent_has_verb(sent):
        return any(map(lambda t: t[1] in {"VB", "VBD", "VBP", "VBZ"}, sent))


    for proposed_sent in compressor.generate_compression_candidates(filterfunc=check_sent_has_verb, n=200):
        plaintext_sent = list(map(lambda x: x[0], proposed_sent))
        lm_score = 1 / (1.0 - lm.estimate_sent_log_proba(plaintext_sent))

        informativeness_score = 0
        for token in proposed_sent:
            #print(token, tr_scores.get(token))
            informativeness_score += tr_scores.get(token, 0)

        score = lm_score * informativeness_score / len(proposed_sent)
        print(plaintext_sent, score, lm_score, informativeness_score, len(proposed_sent))
        sents_and_scores.append((proposed_sent, score))

    sents_and_scores.sort(key=scnd, reverse=True)

    return sents_and_scores[:200]


def select_sentences(per_cluster_candidates, sim_model, maxlen=250):
    sentences_with_index = []

    global_sent_idx = 0
    for cluster_idx, cluster_sents in enumerate(per_cluster_candidates):
        for sent, score in cluster_sents:
            sentences_with_index.append((cluster_idx, global_sent_idx, sent, score))
            global_sent_idx += 1

    p = model('basic')
    select_switch = p.var('p', global_sent_idx, bool)

    maxlen_constraint = None
    max_term = None

    cluster_idx_sums = {}
    cluster_idx_single_member_constraints = {}

    for cluster_idx, global_sent_idx, sent, score in sentences_with_index:
        single_member_constr = cluster_idx_single_member_constraints.get(cluster_idx)

        if single_member_constr is None:
            single_member_constr = select_switch[global_sent_idx]
        else:
            single_member_constr += select_switch[global_sent_idx]

        cluster_idx_single_member_constraints[cluster_idx] = single_member_constr

        if max_term is None:
            max_term = select_switch[global_sent_idx] * score
            maxlen_constraint = select_switch[global_sent_idx] * len(sent)
        else:
            max_term += select_switch[global_sent_idx] * score
            maxlen_constraint += select_switch[global_sent_idx] * len(sent)

        cluster_idx_sum = cluster_idx_sums.get(cluster_idx)
        if cluster_idx_sum is None:
            cluster_idx_sum = select_switch[global_sent_idx]
            cluster_idx_sums[cluster_idx] = cluster_idx_sum
        else:
            cluster_idx_sum += select_switch[global_sent_idx]

        for cluster_idx_2, global_sent_idx_2, sent_2, score_2 in sentences_with_index:
            if global_sent_idx == global_sent_idx_2:
                continue

            if cluster_idx != cluster_idx_2:
                sent_1_toks = list(map(lambda x: x[0], sent))
                sent_2_toks = list(map(lambda x: x[0], sent_2))
                sim = sim_model.compute_similarity(sent_1_toks, sent_2_toks)

                # Why was this commented out?
                if sim > 0.5:
                    select_switch[global_sent_idx] + select_switch[global_sent_idx_2] <= 1.0

    p.maximize(max_term)
    #maxlen_constraint <= maxlen

    for cluster_term in cluster_idx_sums.values():
        cluster_term <= 1.0

    for member_constr in cluster_idx_single_member_constraints.values():
        member_constr <= 1.0

    p.solve()

    sentences = []

    for global_sent_idx, sent_switch in enumerate(select_switch):
        if sent_switch.primal > 0.0:
            cluster_idx, global_sent_idx, sent, score = sentences_with_index[global_sent_idx]
            sentences.append((" ".join([tok for tok, pos in sent]), cluster_idx))

    p.end()

    return sentences


    #sum(score * select_switch[(c_idx, s_idx)] for c_idx, sents in cluster_scores for s_idx, score in sents)


def select_sentences_old(per_cluster_candidates, sim_model, maxlen=250):
    sentences_with_index = [
        ((c_idx, s_idx), sent[0]) for c_idx, sents in enumerate(per_cluster_candidates)
        for s_idx, sent in enumerate(sents)
    ]
    sentences = [sent[1] for sent in sentences_with_index]

    cluster_scores = [
        (c_idx, [(s_idx, sent[1]) for s_idx, sent in enumerate(sents)])
        for c_idx, sents in enumerate(per_cluster_candidates)
    ]
    cluster_indices = [
        (c_idx, s_idx) for c_idx, sents in cluster_scores for s_idx, _ in enumerate(sents)
    ]

    p = model('basic')
    p.verbose(False)
    select_switch = p.var('p', cluster_indices, bool)
    p.maximize(
        sum(score * select_switch[(c_idx, s_idx)] for c_idx, sents in cluster_scores for s_idx, score in sents))

    length_constraint = []
    for c_idx, cluster in cluster_scores:
        cluster_switches = []
        for s_idx, score in cluster:
            cluster_switches.append(select_switch[(c_idx, s_idx)] * 1)

            length_constraint.append(select_switch[(c_idx, s_idx)] * len(per_cluster_candidates[c_idx][s_idx][0]))
        sum(length_constraint) <= maxlen
        sum(cluster_switches) <= 1.0

    for (s_idx_1, (idx_1, sent_1)), (s_idx_2, (idx_2, sent_2)) in it.product(enumerate(sentences_with_index), enumerate(sentences_with_index)):
        if sent_1 == sent_2:
            continue

        if idx_1[0] == idx_2[0]:
            continue

        #sent_1_vec = repr_matrix[s_idx_1]
        #sent_2_vec = repr_matrix[s_idx_2]
#
        #sent_1_vec = sent_1_vec.toarray().reshape(sent_1_vec.shape[1])
        #sent_2_vec = sent_2_vec.toarray().reshape(sent_2_vec.shape[1])
#
        #sim = np.dot(sent_1_vec, sent_2_vec) / (np.linalg.norm(sent_1_vec) * np.linalg.norm(sent_2_vec))

        sent_1_toks = list(map(lambda x: x[0], sent_1))
        sent_2_toks = list(map(lambda x: x[0], sent_2))

        sim = sim_model.compute_similarity(sent_1_toks, sent_2_toks)

        if sim > 0.5:
            select_switch[idx_1] + select_switch[idx_2] <= 1.0

    p.solve()
    #p.solve(int)
    sentences = []
    for idx, switch in enumerate(select_switch):
        if select_switch[switch].primal > 0.0:
            sent = per_cluster_candidates[switch[0]][switch[1]]
            sentences.append((" ".join([tok for tok, pos in sent[0]]), switch[0]))

    p.end()

    return sentences


def calculate_keyword_text_rank(sentences, window_size=None):
    graph = nx.Graph()

    for sent in sentences:
        context = sent
        for tok, pos in sent:
            if tok.lower() in STOPWORDS or not tok.isalnum():
                continue
            graph.add_node((tok, pos))
            for context_tok, context_pos in context:
                if context_tok == tok and context_pos == pos:
                    continue
                if (context_tok.lower() not in STOPWORDS and context_tok.isalnum()):
                    graph.add_edge((context_tok, context_pos), (tok, pos))

    pr = nx.pagerank(graph)

    return pr




def test_sentence_compression():
    compressor = SentenceCompressionGraph(stopwords=STOPWORDS)

    sentences = [
#"The SPD has come to an agreement with the CDU .",
#"The social democratic party of Germany , the SPD , has come to an agreement with the CDU tonight .",
#"SPD and center right CDU agree on coalition talks .",
#"SPD and CDU have come to an agreement on a blueprint for future coalition talks .",
#"Leaders of SPD and CDU have come to an agreement tonight",
        "Anna has bought cookies",
        "Anna Smith has bought chocolat cookies",
        "The chocolat cookies were bought by Anna",
        "Anna bought cookies for the party"
    ]

    pos_tagged = map(lambda s: nltk.pos_tag(s.split()), sentences)

    for idx, sent in enumerate(pos_tagged):
        compressor.add_sentence(sent)
        pydot = to_pydot(compressor.graph)
        pydot.set_rankdir('LR')
        pydot.write("example-{}.dot".format(idx))
        pydot.write_png("example-{}.png".format(idx))

    candidates = compressor.generate_compression_candidates()

    pydot = to_pydot(compressor.graph)
    pydot.set_rankdir('LR')
    #pydot.set_landscape(True)
    print(dir(pydot))
    pydot.write("example.dot")
    pydot.write_png("example.png")

    print(candidates[0:3])


def test_reallife_compression():
    #lm = StoredLanguageModel.from_file("lm_giga_20k_nvp_3gram/lm_giga_20k_nvp_3gram.arpa")
    lm = KenLMLanguageModel.from_file("langmodel20k_vp_3.bin")
    #lm = None

    document_basedir = sys.argv[1]

    for cluster_id in ["331"]:
        documents = []
        cluster_dir = os.path.join(document_basedir, cluster_id)
        for doc_fname in os.listdir(cluster_dir):
            if not doc_fname.endswith(".out"):
                continue
            reader = StanfordXMLReader()
            document = reader.run(os.path.join(cluster_dir, doc_fname))
            documents.append(document)

    noriega_sents = []
    for doc in documents:
        for sent in doc.sentences:
            if "noriega" in sent.as_tokenized_string().lower() or True:
                noriega_sents.append(sent)
    import random
    compressor = SentenceCompressionGraph(STOPWORDS)

    flat_noriega_sents = []

    for sent in noriega_sents:
        flat_noriega_sents.append(sent.as_tokenized_string())

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    tf_idf = TfidfVectorizer("english").fit_transform(flat_noriega_sents)
    sims = cosine_similarity(tf_idf)

    print(sims)

    sim_sents = list(set(map(lambda x: tuple(sorted(x)), filter(lambda x: x[0] != x[1], zip(*np.where(sims > 0.5))))))

    for idx1, idx2 in sim_sents:
        print(noriega_sents[idx1].as_tokenized_string())
        print(noriega_sents[idx2].as_tokenized_string())
        print()

    random.shuffle(noriega_sents)
    noriega_sents = noriega_sents[:15]
    compressor.add_sentences(map(lambda s: s.as_token_tuple_sequence("form", "pos"), noriega_sents))


    print(len(noriega_sents))
    pydot = to_pydot(compressor.graph)
    pydot.write_png("noriega.png")

    generated_sentence = []

    for cnd in compressor.generate_compression_candidates(n=200):
        generated_sentence.append((" ".join(map(lambda x: x[0], cnd)), 
            (1.0 / (1.0 - lm.estimate_sent_log_proba(list(map(lambda x: x[0], cnd)))))))

    generated_sentence.sort(key=lambda t: t[1], reverse=True)

    for s in generated_sentence:
        print(s[0], s[1])


def save_cluster(fname, sentences):
    with open(fname, "w") as f_out:
        for sentence in sentences:
            name = sentence.document.name
            idx = sentence.idx

            f_out.write("{} {}\n".format(name, idx))


def read_cluster_from_premade_files(dirname):
    clusters = {}

    for cl_idx, filename in enumerate(os.listdir(dirname)):
        filepath = os.path.join(dirname, filename)

        if not filepath.endswith(".txt"):
            continue

        clusters[os.path.basename(filename)] = read_cluster_file_banerjee(filepath)

    return clusters


from reader import Token, Sentence


def read_cluster_file_banerjee(path):
    sentences = []

    with open(path, encoding="latin-1") as f:
        for line in f:
            words = line.split()
            tokens = []

            for word in words:
                form, pos = word.split("/")
                tokens.append(Token(form=form, pos=pos))
            sent = Sentence(tokens)
            sent.idx = 0
            sentences.append(sent)

    return sentences


def read_clusters(path, documents):
    document_lookup = dict((doc.name, doc) for doc in documents)

    clusters = dict()

    for fname in iter_files(path, ".txt"):
        basename = os.path.basename(fname)
        clusters[int(basename.split(".")[0])] = read_cluster_file(fname, document_lookup)

    return clusters


def read_cluster_file(path, document_lookup):
    cluster_members = []

    with open(path) as f:
        for line in f:
            docname, sent_id = line.strip().rsplit(" ", 1)
            cluster_members.append(document_lookup[docname].sentences[int(sent_id)])

    return cluster_members


def summ_with_premade_clusters():
    lm = KenLMLanguageModel.from_file("langmodel20k_vp_3.bin")

    clusters = read_cluster_from_premade_files(sys.argv[1])
    outfile = sys.argv[2]

    global_sim_model = SklearnTfIdfCosineSimilarityModel(stem=False)
    global_sim_model.fit([sent.as_token_attr_sequence("form") for cl in clusters.values() for sent in cl])

    candidates = []

    for cluster in clusters.values():
        cl_candidates = []

        tr_scores = calculate_keyword_text_rank([sent.as_token_tuple_sequence("form", "pos") for sent in cluster])

        sim_model = TfidfVectorizer()
        orig_vecs = sim_model.fit_transform([" ".join(sent.as_token_attr_sequence("form")) for sent in cluster])

        def check_closeness(sent):
            vec_1 = sim_model.transform(map(lambda x: x[0], sent))
            similarities = cosine_similarity(vec_1, orig_vecs)

            return all(similarities[0,:] < 0.8)

        word_graph = SentenceCompressionGraph(STOPWORDS)
        word_graph.add_sentences(sent.as_token_tuple_sequence("form", "pos") for sent in cluster)

        for sent in word_graph.generate_compression_candidates(n=200, filterfunc=check_closeness):
            lm_score = 1 / (1.0 - lm.estimate_sent_log_proba(list(map(lambda x: x[0], sent))))
            informativeness_score = calculate_informativeness(sent, tr_scores)
            score = informativeness_score * lm_score / len(sent)

            cl_candidates.append((sent, score))

        candidates.append(cl_candidates)

    #summary = select_sentences_submod(
    #    candidates,
    #    [sent.as_token_tuple_sequence("form", "pos") for sent in cluster for cluster in clusters]
    #)

    summary = select_sentences(candidates, global_sim_model)

    with open(outfile, "w") as f_out:
        for sent in summary:
            f_out.write(sent[0])
            f_out.write("\n")

    print(summary)



if __name__ == "__main__":
    #test_reallife_compression()
    #test_sentence_compression()
    
    main()

    #timeline_main()

    #summ_with_premade_clusters()
