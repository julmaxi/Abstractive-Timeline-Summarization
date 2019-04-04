import tlgraphsum.mturk.db as db
from collections import namedtuple
import re


class Score:
    def __init__(self):
        self.num_correct_answers = 0
        self.num_answers = 0
        self.num_incorrect_answers = 0
        self.num_did_not_know = 0

    def __add__(self, other):
        score = Score()
        score.num_correct_answers = self.num_correct_answers + other.num_correct_answers
        score.num_answers = self.num_answers + other.num_answers
        score.num_incorrect_answers = self.num_incorrect_answers + other.num_incorrect_answers
        score.num_did_not_know = self.num_did_not_know + other.num_did_not_know

        return score

    def __repr__(self):
        return "Score(num_answers={}, num_correct_answers={}, num_incorrect_answers={}, num_did_not_know={})".format(
            self.num_answers,
            self.num_correct_answers,
            self.num_incorrect_answers,
            self.num_did_not_know
        )


def score_answers(questions, answers):
    score = Score()
    for question, answer in zip(questions, answers):
        score.num_answers += 1
        if answer.text is None or answer.user_did_not_find_answer:
            score.num_did_not_know += 1
            continue

        if re.match(question.gold_answer, answer.text):
            score.num_correct_answers += 1
        else:
            score.num_incorrect_answers += 1

    return score


def main():
    db_session = db.open_db("sqlite:///mturk.db")

    all_hits = db_session.query(db.HIT).all()

    total_score = Score()

    for hit in all_hits:
        questions = hit.questions

        for assignment in hit.assignments:
            answers = assignment.answers

            score = score_answers(questions, answers)

            total_score += score

        print(total_score)


if __name__ == "__main__":
    main()
