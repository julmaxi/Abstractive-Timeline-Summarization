import random
import spacy
from collections import Counter, defaultdict
from tilse.data.timelines import Timeline
import sys
import os
from tlgraphsum.utils import iter_dirs, iter_files, ensure_dir_exists

random.seed(1028)


nlp = spacy.load('en_core_web_sm')


def select_dates_to_query(timeline, n=5):
    return random.sample(timeline.get_dates(), n)


def select_entities(timeline):
    master_doc = []
    for summary in timeline.dates_to_summaries.values():
        master_doc.append(" ".join(summary))

    document = nlp(" ".join(master_doc))

    entity_spans = defaultdict(list)

    entity_counter = Counter()
    for entity in document.ents:
        entity_counter[entity.text] += 1
        entity_spans[entity.text].append((entity.start, entity.end))

    question_candidates = []

    for entity, _ in entity_counter.most_common(5):
        sentences = set()
        for start, end in entity_spans[entity]:
            for sentence in document.sents:
                if sentence.start <= start and sentence.end >= end:
                    sentences.add(sentence)
                    break
            else:
                print("No sent found")

        question_candidates.append((entity, sentences))

    return question_candidates


def write_date_questions(topic_gold_tl_dir, gold_tl_fname, tl, db_file, annotator_file):
    date_questions = select_dates_to_query(tl)
    for date in date_questions:
        db_file.write(
            "\t".join([
                os.path.basename(topic_gold_tl_dir),
                os.path.basename(gold_tl_fname),
                "date",
                str(date)
            ])
        )
        db_file.write("\n")
        annotator_file.write("# Ask about when one of the events in the following text happend.\n# {}\n\n".format(" ".join(tl.dates_to_summaries[date])))


def write_entity_questions(topic_gold_tl_dir, gold_tl_fname, tl, db_file, annotator_file):
    entity_questions = select_entities(tl)
    for entity, sentences in entity_questions:
        db_file.write(
            "\t".join([
                os.path.basename(topic_gold_tl_dir),
                os.path.basename(gold_tl_fname),
                "entity",
                entity
            ])
        )
        db_file.write("\n")

        annotator_file.write("# Ask a question about ''{}'' based on one of the following sentences:\n".format(entity))
        for sentence in sentences:
            annotator_file.write("# {} \n".format(sentence))
        annotator_file.write("\n")


def write_entity_detail_question(topic_gold_tl_dir, gold_tl_fname, tl, db_file, annotator_file):
    date_questions = select_dates_to_query(tl)
    date_questions = select_dates_to_query(tl)
    for date in date_questions:
        db_file.write(
            "\t".join([
                os.path.basename(topic_gold_tl_dir),
                os.path.basename(gold_tl_fname),
                "date",
                str(date)
            ])
        )
        db_file.write("\n")
        annotator_file.write("# Ask about when one of the events in the following text happend.\n# {}\n\n".format(" ".join(tl.dates_to_summaries[date])))

        db_file.write(
            "\t".join([
                os.path.basename(topic_gold_tl_dir),
                os.path.basename(gold_tl_fname),
                "detail",
                str(date)
            ])
        )
        db_file.write("\n")
        annotator_file.write("# Ask about a detail of what happened on that day.\n\n")


if __name__ == "__main__":
    gold_tl_dir = sys.argv[1]
    out_file_prefix = sys.argv[2]

    db_file = open(out_file_prefix + ".db", "w")
    annotator_file = open(out_file_prefix + ".txt", "w")

    for topic_gold_tl_dir in iter_dirs(gold_tl_dir):
        for gold_tl_fname in iter_files(topic_gold_tl_dir, ".txt"):
            with open(gold_tl_fname, errors="ignore") as f:
                tl = Timeline.from_file(f)

                #write_entity_questions(topic_gold_tl_dir, gold_tl_fname, tl, db_file, annotator_file)
                #write_date_questions(topic_gold_tl_dir, gold_tl_fname, tl, db_file, annotator_file)
                write_entity_detail_question(topic_gold_tl_dir, gold_tl_fname, tl, db_file, annotator_file)

    db_file.close()
    annotator_file.close()
