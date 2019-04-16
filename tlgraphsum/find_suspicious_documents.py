import os
from tilse.data.timelines import Timeline
import sys
import itertools as it


def min_jaccard_sim(sent_1, sent_2):
    toks_1 = set(sent_1.lower().split())
    toks_2 = set(sent_2.lower().split())

    if len(toks_1) == 0 or len(toks_2) == 0:
        return 0

    return len(toks_1.intersection(toks_2)) / min(len(toks_1), len(toks_2))


def read_tl(fname):
    with open(fname, errors="ignore") as f:
        return Timeline.from_file(f)


def iter_same_timelines(tl_dir_1, tl_dir_2):
    topics1 = set(os.listdir(tl_dir_1))
    topics2 = set(os.listdir(tl_dir_2))

    topics = topics1.intersection(topics2)

    for topic in topics:
        print("_______", topic, "_______")
        topic_dir_1 = os.path.join(tl_dir_1, topic)
        topic_dir_2 = os.path.join(tl_dir_2, topic)

        timelines_1 = set(os.listdir(topic_dir_1))
        timelines_2 = set(os.listdir(topic_dir_2))

        timeline_fnames = timelines_1.intersection(timelines_2)

        for tl_fname in timeline_fnames:
            tl_path_1 = os.path.join(topic_dir_1, tl_fname)
            tl_path_2 = os.path.join(topic_dir_2, tl_fname)

            yield (read_tl(tl_path_1), read_tl(tl_path_2), topic, tl_fname)


def main():
    for tl_1, tl_2, topic, fname in iter_same_timelines(sys.argv[1], sys.argv[2]):
        overlapping_dates = set(tl_1.get_dates()).intersection(set(tl_2.get_dates()))

        for date in overlapping_dates:
            sents_1 = (tl_1.dates_to_summaries[date])
            sents_2 = (tl_2.dates_to_summaries[date])

            for s1, s2 in it.product(sents_1, sents_2):
                if min_jaccard_sim(s1, s2) >= 0.8:
                    print(s1)
                    print(s2)
                    print()




if __name__ == "__main__":
    main()
