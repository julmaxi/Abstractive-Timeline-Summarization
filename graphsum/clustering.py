import logging

import networkx as nx
from collections import namedtuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reader import DateTag


logging.basicConfig(level=logging.DEBUG)

DateIndexEntry = namedtuple("DateIndexEntry", "child_dates members")


def generate_affinity_matrix_from_dated_sentences(sents, threshold=0.2):
    date_sent_index = {}

    for s_idx, sent in enumerate(sents):
        # TODO: How exactly do we deal with undated sentences?
        all_tags = sent.all_date_tags
        if len(all_tags) == 0:
            all_tags = list(sent.document.all_date_tags)
            all_tags.append(sent.document.dct_tag)

        for tag in all_tags:
            year = tag.year
            month = tag.month
            day = tag.day

            year_index_entry = date_sent_index.setdefault(year, DateIndexEntry({}, []))
            year_index_entry.members.append((s_idx, sent))

            if month is not None:
                month_index_entry = year_index_entry.child_dates.setdefault(month, DateIndexEntry({}, []))
                month_index_entry.members.append((s_idx, sent))

                if day is not None:
                    day_index_entry = month_index_entry.child_dates.setdefault(day, DateIndexEntry({}, []))
                    day_index_entry.members.append((s_idx, sent))

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
            logging.info("Processing sent {} of {}".format(s_idx, len(sents)))

        all_tags = sent.all_date_tags
        if len(all_tags) == 0:
            all_tags = list(sent.document.all_date_tags)
            all_tags.append(sent.document.dct_tag)

        connected_sents = []
        for tag in all_tags:
            if tag.dtype == DateTag.DAY:
                pass # handled separatly
                #connected_sents.extend(date_sent_index[year].child_dates[month].child_dates[day].members)
            elif tag.dtype == DateTag.MONTH:
                connected_sents.extend(date_sent_index[year].child_dates[month].members)
            else:  # year only
                connected_sents.extend(date_sent_index[year].members)

        if len(connected_sents) == 0:
            continue

        connected_vecs = vectors[tuple(sid for sid, sent in connected_sents),:]

        sims = cosine_similarity(vectors[s_idx], connected_vecs)
        for (other_s_idx, _), sim in zip(connected_sents, sims[0]):
            if sim >= threshold:
                similarities[s_idx, other_s_idx] = sim

    return similarities


def write_similarity_file(fname, affinities):
    with open(fname, "w") as f_out:
        for (idx1, idx2), sim in affinities.items():
            f_out.write("{} {} {}\n".format(idx1, idx2, sim))

