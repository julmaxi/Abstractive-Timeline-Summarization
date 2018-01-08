from scipy.sparse import lil_matrix, csr_matrix
import scipy.sparse.linalg as sp_linalg

from collections import Counter, defaultdict
from collections.abc import MutableMapping

import numpy as np

import networkx as nx
import itertools as it
import math

def generate_tf_idx_matrix(docset, cutoff=2):
    document_frequencies = Counter()
    total_frequencies = Counter()

    per_doc_frequency_counter = {}

    for d_idx, doc in enumerate(docset):
        doc_frequency_counter = Counter(doc)
        document_frequencies.update(doc_frequency_counter.keys())
        total_frequencies += doc_frequency_counter

        per_doc_frequency_counter[d_idx] = doc_frequency_counter

    words = list(map(lambda x: x[0], filter(lambda x: x[1] >= cutoff, total_frequencies.items())))

    tf_idf_matrix = lil_matrix((len(docset), len(words)))
    for d_idx, doc in enumerate(docset):
        for w_idx, word in enumerate(words):
            tf_idf_matrix[d_idx, w_idx] = per_doc_frequency_counter[d_idx][word] / float(document_frequencies[word])

    return csr_matrix(tf_idf_matrix)


def generate_binary_bow_matrix(docset, cutoff=8, words=None):
    tok_counters = Counter()
    document_token_ids = defaultdict(set)

    for d_idx, doc in enumerate(docset):
        for tok in doc:
            tok_counters[tok] += 1
            document_token_ids[d_idx].add(tok)

    if words is None:
        all_tokens = list(map(lambda x: x[0], filter(lambda x: x[1] >= cutoff, tok_counters.items())))
    else:
        all_tokens = list(words)

    tok_ids = dict(map(lambda x: (x[1], x[0]), enumerate(all_tokens)))

    tf_idf_matrix = lil_matrix((len(docset), len(all_tokens)))
    for d_idx, tokens in document_token_ids.items():
        for tok in tokens:
            tok_id = tok_ids.get(tok)
            if tok_id is None:
                continue
            tf_idf_matrix[d_idx, tok_id] = 1

    return csr_matrix(tf_idf_matrix)


def generate_word_occurence_matrix(docset):
    doc_occ_counter = Counter()
    per_doc_counters = defaultdict(Counter)

    token_indices = {}

    corpus_token_count = 0

    for d_idx, doc in enumerate(docset):
        toks_in_doc = set()
        for tok in doc:
            t_idx = token_indices.get(tok)
            if t_idx is None:
                t_idx = len(token_indices)
                token_indices[tok] = t_idx
            per_doc_counters[d_idx][t_idx] += 1

            toks_in_doc.add(t_idx)

            corpus_token_count += 1

        for t_idx in toks_in_doc:
            doc_occ_counter[t_idx] += 1

    tf_idf_matrix = lil_matrix((len(docset), len(token_indices)))
    for d_idx, counter in per_doc_counters.items():
        for t_idx, cnt in counter.items():
            tf_idf_matrix[d_idx, t_idx] = cnt * (len(docset) / doc_occ_counter[t_idx])

    return csr_matrix(tf_idf_matrix)

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


from sklearn.feature_extraction.text import TfidfVectorizer


class SklearnTfIdfCosineSimilarityModel:
    def __init__(self, *args):
        pass

    def fit(self, documents):
        self.model = TfidfVectorizer()

        self.model.fit(map(lambda d: " ".join(d), documents))

    def compute_similarity(self, sent_1, sent_2):
        sent_1 = " ".join(sent_1)
        sent_2 = " ".join(sent_2)
        vecs = self.model.transform((sent_1, sent_2)).toarray()

        return np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]))


class ModifiedIdfCosineSimilarityModel:
    def __init__(self, stopwords):
        self.document_frequencies = Counter()
        self.stopwords = set(stopwords)
        self.num_docs = 0

    def fit(self, documents):
        for doc in documents:
            tokens = set(map(lambda s: stemmer.stem(s.lower()), filter(lambda t: t not in self.stopwords and t.isalpha(), doc)))
            self.document_frequencies.update(tokens)

            self.num_docs += 1

        print(self.document_frequencies)

    def compute_similarity(self, sent_1, sent_2):
        sent_1 = [t.lower() for t in sent_1]
        sent_2 = [t.lower() for t in sent_2]
        sent_freq_1 = Counter(sent_1)
        sent_freq_2 = Counter(sent_2)

        all_tokens = set(sent_freq_1.keys()) | set(sent_freq_2.keys())

        sent_prod_sum = 0.
        sent_1_sq_sum = 0.
        sent_2_sq_sum = 0.

        for token in all_tokens:
            if token in self.stopwords or not token.isalpha():
                continue
            token = stemmer.stem(token)
            tok_freq_1 = sent_freq_1.get(token, 0)
            tok_freq_2 = sent_freq_2.get(token, 0)

            df = self.document_frequencies.get(token, 0)
            #if df is None:
            #    continue

            idf = math.log((self.num_docs + 1) / (1 + df)) + 1
#
#            if token == "'s":
#                print(idf)
#
#            if token == "gallardo":
#                print("G", idf, df)

            sent_prod_sum += tok_freq_1 * tok_freq_2 * (idf ** 2)
            sent_1_sq_sum += (tok_freq_1 * idf) ** 2
            sent_2_sq_sum += (tok_freq_2 * idf) ** 2

        if sent_1_sq_sum == 0 or sent_2_sq_sum == 0:
            return 0.0

        return sent_prod_sum / (math.sqrt(sent_1_sq_sum) * math.sqrt(sent_2_sq_sum))


class PerKeyDefaultDict(MutableMapping):
    """docstring for PerKeyDefaultDict"""
    def __init__(self, default_factory):
        super(PerKeyDefaultDict, self).__init__()
        self.default_factory = default_factory
        self.store = {}

    def __getitem__(self, key):
        try:
            return self.store[key]
        except KeyError:
            self.store[key] = self.default_factory(key)
            return self.store[key]

    def __setitem__(self, key, val):
        self.store[key] = val

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)



class EmbeddingMatchingCosingSimilarityModel:
    def __init__(self, embeddings):
        self.embeddings = embeddings

        self.pairwise_similarities = PerKeyDefaultDict(self.compute_cosine_dist)

    def compute_cosine_dist(self, word_pair):
        vec_1 = self.embeddings[word_pair[0]]
        vec_2 = self.embeddings[word_pair[1]]

        return vec_1.dot(vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))

    def compute_similarity(self, sent_1, sent_2):
        w_graph = nx.Graph()
        w_graph.add_nodes_from(map(lambda t: (0, (t[0], t[1].lower())), enumerate(sent_1)))
        w_graph.add_nodes_from(map(lambda t: (1, (t[0], t[1].lower())), enumerate(sent_2)))

        for (idx1, t1), (idx2, t2) in it.product(enumerate(sent_1), enumerate(sent_2)):
            if t1 not in self.embeddings or t2 not in self.embeddings:
                continue
            w_graph.add_edge((idx1, t1.lower()), (idx2, t2.lower()), weight=self.pairwise_similarities[t1.lower(), t2.lower()])

        matching = nx.max_weight_matching(w_graph)

        sim = 0.0
        for (idx1, t1), (idx2, t2) in matching.items():
            sim += self.pairwise_similarities[t1, t2]

        return sim / min(len(sent_1), len(sent_2)) # Subsumed sentences should be similar


class EmbeddingsCosingSimilarityModel:
    def __init__(self, embeddings):
        self.embeddings = embeddings

        self.pairwise_similarities = PerKeyDefaultDict(self.compute_cosine_dist)

    def compute_cosine_dist(self, word_pair):
        vec_1 = self.embeddings[word_pair[0]]
        vec_2 = self.embeddings[word_pair[1]]

        return vec_1.dot(vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))

    def compute_similarity(self, sent_1, sent_2):
        matched_indices = set()
        total_sim = 0.0
        for idx_1, (t_1, pos_1) in enumerate(sent_1):
            t_1 = t_1.lower()
            if t_1 not in self.embeddings:
                continue
            best_match_idx = None
            best_match_sim = 0.0
            for idx_2, (t_2, pos_2) in enumerate(sent_2):
                if pos_1 != pos_2:
                    continue
                if t_1 == t_2:
                    matched_indices.add(idx_2)
                    total_sim += 1
                    continue
                t_2 = t_2.lower()
                if t_2 not in self.embeddings:
                    continue
                if idx_2 in matched_indices:
                    continue

                sim = self.pairwise_similarities[t_1.lower(), t_2.lower()]

                if sim > best_match_sim:
                    best_match_sim = sim
                    best_match_idx = idx_2

            if best_match_idx is not None:
                matched_indices.add(best_match_idx)
                total_sim += best_match_sim

        return total_sim / min(len(sent_1), len(sent_2)) # Subsumed sentences should be similar



def read_glove_embeddings(fname):
    embedding_dict = {}
    with open(fname) as f_in:
        for line in f_in:
            key, vecstr = line.strip().split(" ", 1)
            vec = np.fromstring(vecstr, sep=" ")
            embedding_dict[key] = vec

    return embedding_dict
