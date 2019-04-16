import logging

import networkx as nx
from collections import namedtuple, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reader import DateTag
from graphsum import STOPWORDS

import string

PUNCTUATION = set(string.punctuation)

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

DateIndexEntry = namedtuple("DateIndexEntry", "child_dates members exact_date_members")

from subprocess import Popen, PIPE
import subprocess

AP_PATH = "libs/affinity-propagation-sparse/ap"


def cluster_sentences_ap(sents, include_uncertain_date_edges=True, predicted_tag_only=False, untangle_multi_date_sentences=False, post_prune=False):
    if predicted_tag_only:
        ap_matrix = generate_localized_predicted_ap_matrix(sents)
    else:
        if untangle_multi_date_sentences:
            new_sents = []
            sent_date_tags = []
            for sent in sents:
                all_tags = set(sent.all_date_tags)
                all_tags.add(sent.document.dct_tag)
                for tag in all_tags:
                    new_sents.append(sent)
                    sent_date_tags.append({tag})
            sents = new_sents
        else:
            sent_date_tags = []
            for sent in sents:
                all_tags = set(sent.all_date_tags)
                #if len(sent.exact_date_references) == 0: # TODO: maybe revert to "all tags"?
                #            all_tags = list(sent.document.all_date_tags)
                all_tags.add(sent.document.dct_tag)

                sent_date_tags.append(all_tags)

        ap_matrix = generate_affinity_matrix_from_dated_sentences(sents, sent_date_tags, include_uncertain_date_edges=include_uncertain_date_edges)

    logger.debug("Computed {} similarities".format(len(ap_matrix)))

    in_lines = [str(len(sents))]

    for (from_idx, to_idx), sim in ap_matrix.items():
        in_lines.append("{} {} {}".format(from_idx, to_idx, sim))

    output = subprocess.check_output([AP_PATH], input="\n".join(in_lines).encode("utf-8"))

    connection_graph = nx.Graph()

    clusters = defaultdict(list)
    examplars = output.decode("utf-8").split()
    for idx, examplar in enumerate(examplars):
        examplar = int(examplar)
        connection_graph.add_node(idx)
        if idx != examplar:
            #connection_graph[examplar].append(sents[idx])
            connection_graph.add_edge(idx, examplar)

    connected_components = list(nx.connected_components(connection_graph))

    sent_cluster_map = None
    if post_prune:
        cluster_potential_members_map = defaultdict(list)
        sent_potential_cluster_map = defaultdict(list)
        sent_cluster_map = {}

        for cl_idx, sent_ids in enumerate(connected_components):
            for sent_id in sent_ids:
                sent = sents[sent_id]
                sent_potential_cluster_map[sent].append((cl_idx, sent_id))
                cluster_potential_members_map[cl_idx].append(sents[sent_id])

        for sent, cand_clusters in sent_potential_cluster_map.items():
            if len(cand_clusters) == 1:
                sent_cluster_map[sent] = cand_clusters[0][0]
                continue

            max_cl_size = 0
            for cand_cl_id, cand_sent_id in cand_clusters:
                cl_size = len(cluster_potential_members_map[cand_cl_id])
                if cl_size > max_cl_size:
                    max_cl_size = cl_size
                    sent_cluster_map[sent] = cand_cl_id
            print("Selected with max size", max_cl_size, cluster_potential_members_map[sent_cluster_map[sent]])

    clustering = []

    for c_idx, component in enumerate(connected_components):
        cluster_sents = []

        dates = []

        for idx in component:
            if sent_cluster_map is not None and sent_cluster_map.get(sents[idx]) is not None:
                if sent_cluster_map.get(sents[idx], [None]) != c_idx:
                    continue

            if untangle_multi_date_sentences:
                dates.append(list(sent_date_tags[idx])[0])
            cluster_sents.append(sents[idx])

        if len(cluster_sents) == 0:
            continue

        if untangle_multi_date_sentences:
            correct_date = None

            for date in dates:
                if date.dtype == DateTag.DAY:
                    correct_date = date
                    break
            if correct_date is not None:
                clustering.append((cluster_sents, correct_date))
        else:
            clustering.append(cluster_sents)

    logger.info("Found {} clusters".format(len(clustering)))
    return clustering


def generate_localized_predicted_ap_matrix(sents, threshold=0.1):
    similarities = {}

    dated_sentences = defaultdict(list)

    for sidx, sent in enumerate(sents):
        dated_sentences[sent.predicted_date].append((sidx, sent))

    sim_model = TfidfVectorizer()
    all_vecs = sim_model.fit_transform(map(lambda s: s.as_tokenized_string(), sents))

    for sents in dated_sentences.values():
        logger.debug("Computing similarities for {} sentences".format(len(sents)))
        local_vecs = all_vecs[list(map(lambda s: s[0], sents)),:]

        sims = cosine_similarity(local_vecs)

        for idx1, vals in enumerate(sims):
            for idx2, sim in enumerate(vals):
                if idx1 == idx2:
                    continue

                if sim < threshold:
                    continue

                sent_idx_1 = sents[idx1][0]
                sent_idx_2 = sents[idx2][0]

                similarities[sent_idx_1, sent_idx_2] = sim
                similarities[sent_idx_2, sent_idx_1] = sim

    return similarities


def generate_affinity_matrix_from_dated_sentences(sents, sent_date_tags, threshold=0.2, include_uncertain_date_edges=True):
    date_sent_index = {}

    week_index = defaultdict(list)

    for s_idx, (sent, all_tags) in enumerate(zip(sents, sent_date_tags)):

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
                    import datetime
                    try:
                        year, week, _ = datetime.date(year, month, day).isocalendar()
                    except ValueError:
                        logger.warning("Invalid date reference {}-{}-{}".format(year, month, day))
                    else:
                        week_index[year, week].append((s_idx, sent))

    vectors = TfidfVectorizer().fit_transform(map(lambda s: " ".join([tok.form for tok in s if not hasattr(tok, "timex") or tok.timex is None]), sents))
    similarities = {}

    for year_index_entry in date_sent_index.values():
        for month_index_entry in year_index_entry.child_dates.values():
            for day_index_entry in month_index_entry.child_dates.values():
                day_vectors = vectors[tuple(sid for sid, sent in day_index_entry.members),:]
                sims = cosine_similarity(day_vectors)

                for x_idx in range(sims.shape[0]):
                    for y_idx in range(x_idx, sims.shape[1]):
                        if sims[x_idx, y_idx] >= threshold:
                            sim_x_idx = day_index_entry.members[x_idx][0]
                            sim_y_idx = day_index_entry.members[y_idx][0]
                            similarities[sim_x_idx, sim_y_idx] = sims[x_idx, y_idx]
                            similarities[sim_y_idx, sim_x_idx] = sims[x_idx, y_idx]

    logger.debug("Computed {} similarities in dates".format(len(similarities)))

    if not include_uncertain_date_edges:
        return similarities

    for s_idx, sent in enumerate(sents):
        if (s_idx + 1) % 100 == 0:
            logging.info("Processing sent {} of {}".format(s_idx + 1, len(sents)))

        all_tags = set(sent.all_date_tags)
        #if len(all_tags) == 0:
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
                sents_to_add.extend(date_sent_index[tag.year].child_dates[tag.month].exact_date_members)
            elif tag.dtype == DateTag.YEAR:
                sents_to_add.extend(date_sent_index[tag.year].exact_date_members)
            elif tag.dtype == DateTag.WEEK:
                sents_to_add.extend(week_index[tag.year, tag.week])

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

        connected_vecs = vectors[tuple(sid for sid, sent in connected_sents),:]

        sims = cosine_similarity(vectors[s_idx], connected_vecs)

        from scipy.sparse import csr_matrix, find

        _, relevant_indices, _ = find(sims >= threshold)

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
                clusters[representative].extend(clusters[idx])
                clusters[idx] = clusters[representative]
            clusters[idx].append(sentences[idx])

    clustering = []
    for cluster_idx, vals in clusters.items():
        if len(vals) < 2:
            continue
        vals.append(sentences[cluster_idx])
        clustering.append(vals)

    return clustering


def read_ap_file_single(fname, sentences):
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
