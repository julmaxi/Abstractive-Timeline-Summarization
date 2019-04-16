import pickle
from tlsum import compute_temp_tr
import datetime
import sys


def load_tr_scores(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    search_date = datetime.date(*map(int, sys.argv[2].split("-")))
    per_date_scores = load_tr_scores(sys.argv[1])
    smooth_scores = compute_temp_tr(per_date_scores, at_dates=[search_date])

    date_scored_words = list(per_date_scores[search_date].items())
    date_scored_words.sort(key=lambda x: x[1], reverse=True)

    smooth_scored_words = list(smooth_scores[search_date].items())
    smooth_scored_words.sort(key=lambda x: x[1], reverse=True)

    for ((date_tok, date_pos), date_score), ((smooth_tok, smooth_pos), smooth_score) in zip(date_scored_words[:100], smooth_scored_words[:100]):
        print(date_tok, date_pos, date_score, "\t\t\t", smooth_tok, smooth_pos, smooth_score)
