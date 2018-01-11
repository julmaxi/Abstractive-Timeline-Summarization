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
from langmodel import StoredLanguageModel, KenLMLanguageModel
from reader import StanfordXMLReader, Sentence, Token
from similarity import EmbeddingsCosingSimilarityModel, read_glove_embeddings, ModifiedIdfCosineSimilarityModel, SklearnTfIdfCosineSimilarityModel, BinaryCosineSimilarityModel, BinaryOverlapSimilarityModel


def scnd(t):
    return t[1]


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
            elif not form.isalnum():
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
        for src, trg, data in self.graph.edges(data=True):
            if src == "START" or trg == "END":
                data["weight_sl"] = 1.0 / data["frequency"]  # TODO: Check what this should be
                continue

            src_freq = len(self.graph.nodes[src]["mapped_tokens"])
            trg_freq = len(self.graph.nodes[trg]["mapped_tokens"])

            diff = 0.0

            for t_idx_1, _, sent_1 in self.graph.nodes[src]["mapped_tokens"]:
                for t_idx_2, _, sent_2 in self.graph.nodes[trg]["mapped_tokens"]:
                    if sent_1 == sent_2 and t_idx_1 < t_idx_2:
                        diff += (t_idx_2 - t_idx_1) ** -1

            #diff **= -1

            w = (src_freq + trg_freq) / diff  #data["frequency"]

            w /= src_freq * trg_freq

            data["weight_sl"] = w
            data["label"] = str(w)

    def generate_compression_candidates(self, n=200, minlen=8, filterfunc=lambda c: True):
        self.calculate_strong_links_weights()

        candidates = []

        candidate_set = set()

        for path in nx.shortest_simple_paths(self.graph, "START", "END", weight="weight_sl"):
            if len(path) < minlen + 2:  # Account for START and END
                continue

            tokens = [self.graph.nodes[n]["token"] for n in path[1:-1]]

            for form, pos in tokens:
                if pos[0] == "V":
                    break
            else:
                continue

            if not filterfunc(tokens):
                continue

            if tuple(tokens) not in candidate_set:
                candidates.append(tokens)
                candidate_set.add(tuple(tokens))

            if len(candidates) >= n:
                break

        return candidates


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
        summarization = summerize_documents(documents, lm)

        with open("{}.sum.txt".format(cluster_id), "w") as f_out:
            f_out.write(summarization)


def timeline_main():
    lm = KenLMLanguageModel.from_file("langmodel20k_vp_3.bin")
    document_basedir = sys.argv[1]

    best_date_id = None
    best_num_dates = None

    for date_id in os.listdir(document_basedir):
        date_path = os.path.join(document_basedir, date_id)
        num_dates = len(os.listdir(date_path))

        if best_date_id is None or num_dates > best_num_dates:
            best_date_id = date_id
            best_num_dates = num_dates


    best_date_id = "2011-03-15"

    documents = []

    best_date_path = os.path.join(document_basedir, date_id)

    for fname in os.listdir(best_date_path):
        try:
            reader = StanfordXMLReader()
            document = reader.run(os.path.join(best_date_path, fname))
            documents.append(document)
        except:
            pass

    summarization = summerize_documents(documents, lm)

    print(summarization)


def summerize_documents(documents, lm):
    #embeddings_similarity_model = EmbeddingsCosingSimilarityModel(read_glove_embeddings("glove.6B.50d.txt"))
    #similarity_model = ModifiedIdfCosineSimilarityModel(STOPWORDS)
    #similarity_model = BinaryCosineSimilarityModel(STOPWORDS)
    #similarity_model.fit([doc.as_token_attr_sequence("form") for doc in documents])

    similarity_model = BinaryOverlapSimilarityModel(STOPWORDS)
    similarity_model.fit([sent.as_token_attr_sequence("form") for doc in documents for sent in doc])

    best_doc = documents[rank_documents_docset_sim(documents, similarity_model)]

    #similarity_model = BinaryCosineSimilarityModel(STOPWORDS)
    #similarity_model = BinaryOverlapSimilarityModel(STOPWORDS)

    clusters = cluster_with_seed_sentences(best_doc.sentences, documents, similarity_model)

    #clusters = {0: (None, [doc.sentences[0] for doc in documents])}

    for cluster_id, (seed, members) in list(clusters.items()):
        print(cluster_id)
        for member in members:
            print(member.as_tokenized_string(), (1.0 / (1.0 - lm.estimate_sent_log_proba(member.as_token_attr_sequence("form")))))
        #if len(members) < len(documents) / 2:
        if len(members) < 2:
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
        candidates = generate_summary_candidates(list(map(lambda s: s.as_token_tuple_sequence("form_lowercase", "pos"), cluster_sentences)), lm)
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


def generate_summary_candidates(sentences, lm):
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

    for proposed_sent in compressor.generate_compression_candidates(filterfunc=check_sent_has_verb):
        plaintext_sent = list(map(lambda x: x[0], proposed_sent))
        lm_score = 1 / (1.0 - lm.estimate_sent_log_proba(plaintext_sent))

        informativeness_score = 0
        for token in proposed_sent:
            print(token, tr_scores.get(token))
            informativeness_score += tr_scores.get(token, 0)

        score = lm_score * informativeness_score / len(proposed_sent)
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
        print(plaintext_sent)
        lm_score = 1 / (1.0 - lm.estimate_sent_log_proba(plaintext_sent))

        informativeness_score = 0
        for token in proposed_sent:
            #print(token, tr_scores.get(token))
            informativeness_score += tr_scores.get(token, 0)

        score = lm_score * informativeness_score / len(proposed_sent)
        sents_and_scores.append((proposed_sent, score))

    sents_and_scores.sort(key=scnd, reverse=True)

    return sents_and_scores[:200]


def select_sentences(per_cluster_candidates, sim_model, maxlen=250):
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
        for tok in sent:
            if tok[0] in STOPWORDS or not tok[0].isalnum():
                continue
            graph.add_node(tok)
            for context_tok in context:
                if context_tok == tok:
                    continue
                if context_tok[0] not in STOPWORDS and tok[0].isalnum():
                    graph.add_edge(context_tok, tok)
            #context.append(tok)

    pr = nx.pagerank(graph)

    return pr




def test_sentence_compression():
    compressor = SentenceCompressionGraph(stopwords=STOPWORDS)

    sentences = [
        "General Antonia Noriega 's defence gained traction",
        "General Antonia Noriega 's defence has gained traction",
        "General Antonia Noriega 's defence is improving",
        "Gen. Noriega 's attorney said he was not guilty",
        "' This is not true ' , said the prosecutor",
        "' In fact it it quite wrong ' , said the state attorney"
    ]

    pos_tagged = map(lambda s: nltk.pos_tag(s.split()), sentences)

    compressor.add_sentences(pos_tagged)

    #compressor.add_sentence(nltk.pos_tag("US President George W. Bush visits the Greenfield Memorial .".split()))
    #compressor.add_sentence(nltk.pos_tag("George W. Bush , who was sworn in last sunday , visits the Greenfield Memorial .".split()))
    #compressor.add_sentence(nltk.pos_tag("Bush is sworn in this sunday .".split()))
    #compressor.add_sentence(nltk.pos_tag("Bush visits us .".split()))
#
    #compressor.add_sentence(nltk.pos_tag("'' We are happy about this opportunity '' , says Greenfield Mayor Tom Smith".split()))
#
    #compressor.add_sentence(nltk.pos_tag("'' I hate this '' , says his son".split()))

    candidates = compressor.generate_compression_candidates()

    pydot = to_pydot(compressor.graph)
    pydot.write("out.dot")
    pydot.write_png("out.png")


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

    for cnd in compressor.generate_compression_candidates(n=50):
        generated_sentence.append((" ".join(map(lambda x: x[0], cnd)), 
            (1.0 / (1.0 - lm.estimate_sent_log_proba(list(map(lambda x: x[0], cnd)))))))

    generated_sentence.sort(key=lambda t: t[1], reverse=True)

    for s in generated_sentence:
        print(s[0], s[1])


if __name__ == "__main__":
    #test_reallife_compression()
    #test_sentence_compression()
    #main()

    timeline_main()
