from collections import Counter
import datetime


def compute_relative_date_frequencies(corpus):
    dateref_counter = Counter()

    num_sents = 0

    for doc in corpus:
        dateref_counter[datetime.date(doc.dct_tag.year, doc.dct_tag.month, doc.dct_tag.day)]  +=  1
        for sent in doc:
            for date_ref in sent.exact_date_references:
                try:
                    dateref_counter[datetime.date(date_ref.year, date_ref.month, date_ref.day)] += 1
                except ValueError:
                    pass
            num_sents += 1

    return dict((date_ref, score / num_sents) for date_ref, score in dateref_counter.items())
