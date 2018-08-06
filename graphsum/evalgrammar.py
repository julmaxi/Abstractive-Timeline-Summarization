import spacy
import sys
import os

from tilse.data.timelines import Timeline


def compute_stats_for_tl(tlfname):
    num_splitted_sents = 0
    root_counts = []

    with open(tlfname, errors="ignore") as f:
        timeline = Timeline.from_file(f)

    for date in timeline:
        for sent in timeline[date]:
            sents = nlp(sent)
            if len(sents) > 1:
                num_splitted_sents += 1

            for sent in sents:
                num_roots = 0
                for tok in sents:
                    if tok.head == tok:
                        num_roots += 1
                root_counts.append(num_roots)

    return num_splitted_sents, root_counts


if __name__ == "__main__":
    nlp = spacy.load("en")

    num_splitted_sents = 0
    root_counts = []

    for topic_name in os.listdir(sys.argv[1]):
        topic_dir = os.path.join(sys.argv[1], topic_name)
        for tl_name in os.listdir(topic_dir):
            if not tl_name.endswith(".txt"):
                continue
            tl_file = os.path.join(topic_dir, tl_name)

            new_num_splitted_sents, new_root_counts = compute_stats_for_tl(tl_file)

            num_splitted_sents += new_num_splitted_sents
            root_counts.extend(new_root_counts)

    print(num_splitted_sents)
    print(sum(root_counts) / len(root_counts))
