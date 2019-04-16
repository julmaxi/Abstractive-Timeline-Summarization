from collections import namedtuple
from tlgraphsum.utils import iter_files, iter_dirs, fst, scnd
from tilse.data.timelines import Timeline


TimelineParameters = namedtuple("TimelineParameters", "first_date last_date max_date_count max_sent_count max_token_count max_date_sent_count")


def determine_tl_parameters(timeline, use_average=True, use_token_count=True):
    dateset = timeline.get_dates()
    earliest_date = min(*dateset)
    last_date = max(*dateset)
    tl_date_count = len(dateset)

    date_sent_counts = []

    date_sent_lens = []

    max_date_sent_count = 0
    total_sent_len = 0
    for date in timeline:
        sents = timeline[date]
        total_sent_len += len(sents)

        max_date_sent_count = max(max_date_sent_count, len(sents))
        date_sent_counts.append(len(sents))

        date_sent_lens.append(sum(len(sent.split()) for sent in sents))

    date_sent_count = None
    date_token_count = None

    if use_average:
        date_sent_count = int(sum(date_sent_counts) / len(date_sent_counts))
        date_token_count = int(sum(date_sent_lens) / len(date_sent_lens))
    else:
        date_sent_count = max_date_sent_count
        date_token_count = max(date_sent_lens)

    if not use_token_count:
        date_token_count = None

#    print(TimelineParameters(
#        earliest_date,
#        last_date,
#        tl_date_count,
#        total_sent_len,
#        date_token_count,
#        date_sent_count
#    ))

    return TimelineParameters(
        earliest_date,
        last_date,
        tl_date_count,
        total_sent_len,
        date_token_count,
        date_sent_count
    )
