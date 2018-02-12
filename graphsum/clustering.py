import networkx as nx
from collections import namedtuple

DateIndexEntry = namedtuple("DateIndexEntry", "child_dates members")

def generate_affinity_matrix_from_dated_sentences(sents, similarity_measure):
    print(len(sents))

    date_sent_index = {}

    for s_idx, sent in enumerate(sents):
        # TODO: How exactly do we deal with undated sentences?
        all_tags = sent.all_date_tags
        if len(all_tags) == 0:
            all_tags = list(sent.document.all_date_tags)
            all_tags.append(sent.document.dct_tag)

        for year, month, day in all_tags:
            year_index_entry = date_sent_index.setdefault(year, DateIndexEntry({}, []))
            year_index_entry.members.append((s_idx, sent))

            if month is not None:
                month_index_entry = year_index_entry.child_dates.setdefault(month, DateIndexEntry({}, []))
                month_index_entry.members.append((s_idx, sent))

                if day is not None:
                    day_index_entry = month_index_entry.child_dates.setdefault(day, DateIndexEntry({}, []))
                    day_index_entry.members.append((s_idx, sent))

    similarities = {}

    for s_idx, sent in enumerate(sents):
        all_tags = sent.all_date_tags
        if len(all_tags) == 0:
            all_tags = list(sent.document.all_date_tags)
            all_tags.append(sent.document.dct_tag)

        for year, month, day in all_tags:
            if day is not None:
                connected_sents = date_sent_index[year].child_dates[month].child_dates[day].members
            elif month is not None:
                connected_sents = date_sent_index[year].child_dates[month].members
            #else:  # year only
            #    connected_sents = date_sent_index[year].members

        print(len(connected_sents))

        for other_s_idx, other_sent in connected_sents:
            sim = similarity_measure(sent.as_token_attr_sequence("form"), other_sent.as_token_attr_sequence("form"))

            similarities[s_idx, other_s_idx] = sim

    return similarities


def write_similarity_file(fname, affinities):
    with open(fname, "w") as f_out:
        for (idx1, idx2), sim in affinities.items():
            f_out.write("{} {} {}\n".format(idx1, idx2, sim))

