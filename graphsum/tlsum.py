from tilse.data import timelines
from tilse.evaluation import rouge

import os

from utils import iter_files, iter_dirs, fst, scnd

from collections import namedtuple, Counter, defaultdict
import sys
import datetime

from graphsum import summarize_timeline_dir

from langmodel import KenLMLanguageModel
from reader import DatedSentenceReader

TimelineParameters = namedtuple("TimelineParameters", "first_date last_date max_date_count max_sent_count max_date_sent_count")


def determine_parameters(gold_dir):
    global_earliest_date = None
    global_last_date = None

    max_tl_date_count = 0

    max_tl_date_sent_count = 0
    max_tl_sent_count = 0

    all_timelines = []
    for fpath in iter_files(gold_dir, ".txt"):
        with open(fpath, encoding="latin-1") as f:
            timeline = timelines.Timeline.from_file(f)
        all_timelines.append(timeline)

        dateset = timeline.get_dates()
        earliest_date = min(*dateset)
        if global_earliest_date is None or earliest_date < global_earliest_date:
            global_earliest_date = earliest_date
        last_date = max(*dateset)
        if global_last_date is None or global_last_date > global_last_date:
            global_last_date = last_date

        max_tl_date_count = max(len(dateset), max_tl_date_count)

        total_sent_len = 0
        for date in timeline:
            sents = timeline[date]
            total_sent_len += len(sents)

            max_tl_date_sent_count = max(max_tl_date_sent_count, len(sents))

        max_tl_sent_count = max(total_sent_len, max_tl_sent_count)

    return TimelineParameters(
        global_earliest_date,
        global_last_date,
        max_tl_date_count,
        max_tl_sent_count,
        max_tl_date_sent_count)


def run_full_tl_summ(timeline_func):
    params = determine_parameters(sys.argv[2])
    tl = timeline_func(sys.argv[1], params)
    with open("timeline.txt", "w") as f_out:
        f_out.write(str(tl))


def select_best_date_by_doc_freq(max_date_count):
    doc_counter = Counter()
    for dir_ in iter_dirs(document_dir):
        date = date_from_dirname(os.path.basename(dir_))
        if date is None:
            continue
        if date <= parameters.last_date and date >= parameters.first_date:
            print(dir_)
            doc_counter[date] = len(list(iter_files(dir_, "cont")))

    dates = list(map(fst, doc_counter.most_common(max_date_count)))
    return dates


def create_timeline(document_dir, parameters):
    dates = select_best_date_by_doc_freq(parameters.max_date_count)

    date_summary_dict = {}

    lm = KenLMLanguageModel.from_file("langmodel20k_vp_3.bin")
    for date in dates:
        date_path = os.path.join(document_dir, date.strftime("%Y-%m-%d"))
        summarization = summarize_timeline_dir(date_path, lm)

        if len(summarization.strip()) > 0:
            date_summary_dict[date] = summarization.split("\n")

    return timelines.Timeline(date_summary_dict)


def date_from_dirname(dirname):
    parts = dirname.split("-")
    assert len(parts) == 3, "Dirname must have format YYYY-mm-dd"
    year, month, day = parts
    if int(day) == 0:
        return None

    return datetime.date(int(year), int(month), int(day))


def create_timeline_sentence_level(document_dir, parameters):
    reader = DatedSentenceReader()

    sents_by_date = defaultdict(list)

    for date_dir in iter_dirs(document_dir):
        print("Reading", date_dir)
        dir_date = datetime.datetime.strptime(os.path.basename(date_dir), "%Y-%m-%d")

        for timeml_fname in iter_files(date_dir, "timeml"):
            basename = timeml_fname[:-7]
            sentences = reader.read(basename, timeml_fname, dir_date)

            for sent in sentences:
                sents_by_date[sent.predicted_date].append(sent)

    print(len(sents_by_date))
    print(sum(map(lambda s: len(s), sents_by_date.values())))

    #dates = select_best_date_by_doc_freq(parameters.max_date_count)



    #date_summary_dict = {}

    lm = KenLMLanguageModel.from_file("langmodel20k_vp_3.bin")


from clustering import generate_affinity_matrix_from_dated_sentences, write_similarity_file
from similarity import SklearnTfIdfCosineSimilarityModel

def create_timeline_clustering(document_dir, parameters):
    reader = DatedSentenceReader()

    all_sents = []
    all_doc_texts = []

    for date_dir in iter_dirs(document_dir):
        dir_date = datetime.datetime.strptime(os.path.basename(date_dir), "%Y-%m-%d")

        for timeml_fname in iter_files(date_dir, "timeml"):
            basename = timeml_fname[:-7]
            doc = reader.read(basename, timeml_fname, dir_date)

            all_doc_texts.append(doc.as_token_attr_sequence("form"))

            for sent in doc.sentences:
                all_sents.append(sent)


    sim_model = SklearnTfIdfCosineSimilarityModel(stem=False)
    sim_model.fit(all_doc_texts)

    affinities = generate_affinity_matrix_from_dated_sentences(all_sents, sim_model)
    write_similarity_file("similarities.txt", affinities)

if __name__ == "__main__":
    run_full_tl_summ(create_timeline_clustering)
