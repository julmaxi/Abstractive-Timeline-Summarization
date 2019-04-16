from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reader import StanfordXMLReader
import sys
from tilse.data import timelines

from utils import iter_dirs, iter_files

import numpy as np


def eval_distance_to_orig():
    all_sents = None
    with open(sys.argv[1]) as f:
        timeline = timelines.Timeline.from_file(f)
        all_sents = list(sent for date in timeline for sent in timeline[date])

    docs = []

    reader = StanfordXMLReader()
    for dirname in iter_dirs(sys.argv[2]):
        for filename in iter_files(dirname, ".htm.cont.tokenized"):
            try:
                docs.append(reader.run(filename))
            except:
                pass


    tfidf = TfidfVectorizer(stop_words=None)
    tfidf.fit(map(lambda d: d.plaintext, docs))

    sent_vecs = tfidf.transform(map(lambda s: " ".join(s.as_token_attr_sequence("form")), [sent for doc in docs for sent in doc.sentences]))
    tl_vecs = tfidf.transform(all_sents)

    sims = cosine_similarity(tl_vecs, sent_vecs)

    max_sims = np.max(sims, 1)

    print(max_sims)
    print("Median", np.median(max_sims))
    print("Min", np.min(max_sims))
    print(all_sents[np.argmin(max_sims)])
    print("Max", np.max(max_sims))


if __name__ == "__main__":
    eval_distance_to_orig()
