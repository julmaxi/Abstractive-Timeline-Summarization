import sys
from tilse.data.timelines import Timeline
import os
import shutil
from collections import deque
import math


def split_to_maxlen(text, maxlen):
    num_splits = int(math.ceil(len(text) / maxlen))

    for split in range(num_splits):
        start = split * maxlen
        end = (split + 1) * maxlen

        yield text[start:end]


def iter_tokenized_text(text, maxlen):
    current_buffer = []
    curr_buffer_size = 0
    to_process = deque(text.split())
    while len(to_process) > 0:
        token = to_process.popleft()
        if len(token) > maxlen:
            toks = list(split_to_maxlen(token, maxlen - 1))
            to_process.extendleft([t + "-" for t in toks] + toks[-1:])
            token = to_process.popleft()

        if (len(token) + curr_buffer_size) <= maxlen:
            current_buffer.append(token)
            curr_buffer_size += len(token) + 1  # One for the space
        else:
            yield " ".join(current_buffer)
            curr_buffer_size = len(token)
            current_buffer.clear()
            current_buffer.append(token)

    if curr_buffer_size > 0:
        yield " ".join(current_buffer)


def print_vertical_split(*texts):
    print("\n".join(format_vertical_split(*texts)))


def format_vertical_split(*texts):
    divider_width = 1
    columns = shutil.get_terminal_size((80, 20)).columns

    max_line_width = (columns - divider_width * (len(texts) - 1)) // (len(texts))

    lines = []

    token_iters = [iter_tokenized_text(text, max_line_width - 1) for text in texts]

    while True:
        text_passages = [next(it, None) for it in token_iters]
        if all(p is None for p in text_passages):
            break

        line = []
        for passage in text_passages:
            if passage is None:
                passage = ""

            if len(passage) < max_line_width:
                passage += " " * (max_line_width - len(passage))

            line.append(passage)

        yield "|".join(line)


def main():
    gold_tl_fname = sys.argv[1]
    sys_tl_fname = sys.argv[2]

    with open(gold_tl_fname, errors="ignore") as f:
        gold_tl = Timeline.from_file(f)

    with open(sys_tl_fname, errors="ignore") as f:
        sys_tl = Timeline.from_file(f)

    all_dates = sorted(set(gold_tl.get_dates()).union(set(sys_tl.get_dates())))

    for date in all_dates:
        print(date)
        if date in gold_tl.dates_to_summaries:
            gold_sum = "\n".join(gold_tl.dates_to_summaries[date])
        else:
            gold_sum = "-----"

        if date in sys_tl.dates_to_summaries:
            sys_sum = "\n".join(sys_tl.dates_to_summaries[date])
        else:
            sys_sum = "-----"
        print_vertical_split(gold_sum, sys_sum)
        print()


if __name__ == "__main__":
    main()