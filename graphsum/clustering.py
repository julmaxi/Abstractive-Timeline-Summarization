import logging

import networkx as nx
from collections import namedtuple, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reader import DateTag
from graphsum import STOPWORDS

import string

PUNCTUATION = set(string.punctuation)

logging.basicConfig(level=logging.INFO)

DateIndexEntry = namedtuple("DateIndexEntry", "child_dates members exact_date_members")

from subprocess import Popen, PIPE
import subprocess

AP_PATH = "libs/affinity-propagation-sparse/ap"


def cluster_sentences_ap(sents):
    ap_matrix = generate_affinity_matrix_from_dated_sentences(sents)

    in_lines = [str(len(sents))]

    for (from_idx, to_idx), sim in ap_matrix.items():
        in_lines.append("{} {} {}".format(from_idx, to_idx, sim))

    output = subprocess.check_output([AP_PATH], input="\n".join(in_lines).encode("utf-8"))

    clusters = defaultdict(list)
    examplars = output.decode("utf-8").split()
    for idx, examplar in enumerate(examplars):
        examplar = int(examplar)
        if idx != examplar:
            clusters[examplar].append(sents[idx])

    clustering = []
    for cluster_idx, vals in clusters.items():
        vals.append(sents[cluster_idx])
        clustering.append(vals)

    return clustering



def generate_affinity_matrix_from_dated_sentences(sents, threshold=0.01):
    date_sent_index = {}

    for s_idx, sent in enumerate(sents):
        # TODO: How exactly do we deal with undated sentences?
        all_tags = set(sent.all_date_tags)
        if len(all_tags) == 0:
#            all_tags = list(sent.document.all_date_tags)
            all_tags.add(sent.document.dct_tag)

        for tag in all_tags:
            year = tag.year
            month = tag.month
            day = tag.day

            year_index_entry = date_sent_index.setdefault(year, DateIndexEntry({}, [], []))
            year_index_entry.members.append((s_idx, sent))

            if month is not None:
                month_index_entry = year_index_entry.child_dates.setdefault(month, DateIndexEntry({}, [], []))
                month_index_entry.members.append((s_idx, sent))

                if day is not None:
                    day_index_entry = month_index_entry.child_dates.setdefault(day, DateIndexEntry({}, [], []))
                    day_index_entry.members.append((s_idx, sent))
                    year_index_entry.exact_date_members.append((s_idx, sent))
                    month_index_entry.exact_date_members.append((s_idx, sent))
                    day_index_entry.exact_date_members.append((s_idx, sent))

    vectors = TfidfVectorizer().fit_transform(map(lambda s: s.as_tokenized_string(), sents))
    similarities = {}

    for year_index_entry in date_sent_index.values():
        for month_index_entry in year_index_entry.child_dates.values():
            for day_index_entry in month_index_entry.child_dates.values():
                day_vectors = vectors[tuple(sid for sid, sent in day_index_entry.members),:]
                sims = cosine_similarity(day_vectors)

                for x_idx in range(sims.shape[0]):
                    for y_idx in range(x_idx, sims.shape[1]):
                        if sims[x_idx, y_idx] >= threshold:
                            similarities[x_idx, y_idx] = sims[x_idx, y_idx]
                            similarities[y_idx, x_idx] = sims[x_idx, y_idx]

    for s_idx, sent in enumerate(sents):
        if (s_idx + 1) % 100 == 0:
            logging.info("Processing sent {} of {}".format(s_idx + 1, len(sents)))

        all_tags = set(sent.all_date_tags)
        if len(all_tags) == 0:
            #all_tags = set(sent.document.all_date_tags)
            all_tags.add(sent.document.dct_tag)
        #print(all_tags, sent.as_tokenized_string())

        connected_sents = []
        sents_to_add = []
        for tag in all_tags:
            if tag.dtype == DateTag.DAY:
                pass # handled separatly
                #connected_sents.extend(date_sent_index[tag.year].child_dates[tag.month].child_dates[tag.day].members)
            elif tag.dtype == DateTag.MONTH:
                sents_to_add = date_sent_index[tag.year].child_dates[tag.month].exact_date_members
            elif tag.dtype == DateTag.YEAR:
                sents_to_add = date_sent_index[tag.year].exact_date_members

        tok_set = set(filter(lambda t: t.lower() not in STOPWORDS and not all(map(lambda c: c in PUNCTUATION, t)), sent.as_token_attr_sequence("form_lowercase")))
        for sidx, other_sent in sents_to_add:
            num_matches = 0
            for tok in other_sent:
                if tok.form in tok_set:
                    num_matches += 1

                if num_matches >= 2:
                    break
            else:
                continue

            connected_sents.append((sidx, other_sent))

        if len(connected_sents) == 0:
            continue

        #print(len(connected_sents))
        #print(sent.as_tokenized_string())
        #print(connected_sents[0][1].as_tokenized_string())
        #print(set(connected_sents[0][1].as_tokenized_string().split()) & tok_set)
        #print("-" * 10)

        #print(len(connected_sents), all_tags, sent.as_tokenized_string(), sent.all_date_tags)
        #print("-" * 10)
        #print(sent.as_tokenized_string())
        #print("\n".join(map(lambda s: s[1].as_tokenized_string(), connected_sents)))
        #print("-" * 10)

        connected_vecs = vectors[tuple(sid for sid, sent in connected_sents),:]

        sims = cosine_similarity(vectors[s_idx], connected_vecs)

        from scipy.sparse import csr_matrix, find

        _, relevant_indices, _ = find(sims >= threshold)

        for other_s_idx in relevant_indices:
            similarities[s_idx, other_s_idx] = sims[0, other_s_idx]

        for (other_s_idx, other_sent), sim in zip(connected_sents, sims[0]):
            if sim >= threshold:
                similarities[s_idx, other_s_idx] = sim

    return similarities


def write_similarity_file(fname, num_sents, affinities):
    with open(fname, "w") as f_out:
        f_out.write(str(num_sents))
        f_out.write("\n")
        for (idx1, idx2), sim in affinities.items():
            f_out.write("{} {} {}\n".format(idx1, idx2, sim))


def read_ap_file(fname, sentences):
    clusters = defaultdict(list)
    with open(fname) as f:
        representatives = f.read().strip().split()

        for idx, representative in enumerate(representatives):
            representative = int(representative)
            if idx != representative:
                clusters[representative].append(sentences[idx])

    clustering = []
    for cluster_idx, vals in clusters.items():
        vals.append(sentences[cluster_idx])
        clustering.append(vals)

    return clustering
