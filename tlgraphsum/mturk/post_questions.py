from jinja2 import Template
import os
import sys
import argparse
from tilse.data.timelines import Timeline
from collections import defaultdict, namedtuple
import spacy
import boto3
import tlgraphsum.mturk.db as db
from xml.etree import ElementTree as ET
import time

Question = namedtuple("Question", "text type answer")

nlp = spacy.load('en_core_web_sm')

with open(os.path.join(os.path.dirname(__file__), "questiontemplate.html")) as file_:
    question_template = Template(file_.read())


SANDBOX_ENDPOINT = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"


class MTurkClient:
    QUESTION = """
<HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
  <HTMLContent><![CDATA[
  {html}
  ]]></HTMLContent>
  <FrameHeight>0</FrameHeight>
</HTMLQuestion>
"""

    def __init__(self, db_session=None, sandbox=True):
        if db_session is None:
            db_session = db.open_db("sqlite:///mturk.db")
        self.db_session = db_session
        args = {}
        if sandbox:
            args["endpoint_url"] = SANDBOX_ENDPOINT
        self.client = boto3.client('mturk', **args)

        self._hittype = None

    def create_hittype(self):
        response = self.client.create_hit_type(
            AutoApprovalDelayInSeconds=60 * 60 * 24,
            AssignmentDurationInSeconds=60 * 60,
            Reward='0.01',
            Title='Readability' + str(int(time.time()))[-10:],
            Keywords='nlp',
            Description='Test',
            QualificationRequirements=[]
        )

        return response["HITTypeId"]

    @property
    def hittype(self):
        if self._hittype:
            return self._hittype.id
        hit_type_id = self.create_hittype()
        self._hittype = self.db_session.query(db.HITType).get(hit_type_id)
        if not self._hittype:
            self._hittype = db.HITType(id=hit_type_id)
            self.db_session.add(self._hittype)
            self.db_session.commit()
        return self._hittype.id

    def upload_batch(self, values):
        for value in values:
            identifier = value["topic"] + "__" + value["timeline_name"]
            hit = db.HIT()
            hit.topic_name = value["topic"]
            hit.timeline_name = value["timeline_name"]
            hit.system_name = value["system"]
            hit.type_id = self.hittype

            for question in value["questions"]:
                db_question = db.Question()
                db_question.text = question.text
                db_question.gold_answer = question.answer
                db_question.type = question.type

                hit.questions.append(db_question)

            response = self.client.create_hit_with_hit_type(
                HITTypeId=self.hittype,
                MaxAssignments=2,
                LifetimeInSeconds=60 * 60 * 24 * 14,
                Question=self.QUESTION.format(html=value["html"]),
                RequesterAnnotation=identifier)

            hit.id = response["HIT"]["HITId"]
            self.db_session.add(hit)
            self.db_session.commit()

    def fetch_multipage_operation(self, method, params={}, collector=lambda i: i, flatten=True):
        next_token = None
        all_results = []
        while True:
            if next_token is not None:
                response = method(
                    NextToken=next_token,
                    **params
                )
            else:
                response = method(
                    **params
                )

            if flatten:
                all_results.extend(collector(response))
            else:
                all_results.append(collector(response))

            next_token = response.get("NextToken")

            if next_token is None:
                break

        return all_results

    def batch_approve_hits(self):
        reviewable_hit_ids = []

        reviewable_hit_ids = self.fetch_multipage_operation(self.client.list_reviewable_hits, collector=lambda r: map(lambda h: h["HITId"], r["HITs"]))

        for hit_id in reviewable_hit_ids:
            hit = self.db_session.query(db.HIT).get(hit_id)
            if not hit:
                continue
            self.review_hit(hit)

    def review_hit(self, hit):
        response = self.fetch_multipage_operation(
            self.client.list_assignments_for_hit,
            {"HITId": hit.id, "AssignmentStatuses": ["Submitted"]},
            collector=lambda r: r["Assignments"]
        )

        ns = {'mturk': "http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/QuestionFormAnswers.xsd"}

        for assignment in response:
            answer_xml = assignment["Answer"]
            tree = ET.fromstring(answer_xml)
            answer_dict = defaultdict(dict)
            start_time = None
            end_time = None
            for answer in tree.findall(".//mturk:Answer", ns):
                question_id = answer.find("mturk:QuestionIdentifier", ns).text
                answer_text = answer.find("mturk:FreeText", ns).text

                if question_id == "startTime":
                    start_time = int(answer_text)
                elif question_id == "endTime":
                    end_time = int(answer_text)
                else:
                    qid_parts = question_id.split("-")
                    question_index = int(qid_parts[1])
                    if len(qid_parts) == 2:
                        answer_dict[question_index]["text"] = answer_text
                    elif qid_parts[2] == "time" and answer_text:
                        answer_dict[question_index]["timestamp"] = int(answer_text)
                    elif qid_parts[2] == "donotknow":
                        answer_dict[question_index]["user_did_not_find_answer"] = answer_text == "on"

            db_assignment = db.HITAssignment()
            db_assignment.id = assignment["AssignmentId"]
            db_assignment.mturk_worker_id = assignment["WorkerId"]
            #db_assignment.answers = [it[1] for it in sorted(answer_dict.items())]
            db_assignment.hit_id = hit.id
            db_assignment.start_time = start_time
            db_assignment.end_time = end_time

            for question_index, answer_info in answer_dict.items():
                with self.db_session.no_autoflush:
                    db_answer = db.WorkerAnswer()
                    db_answer.question_id = hit.questions[question_index].id
                    db_answer.assignment_id = db_assignment.id
                    db_answer.timestamp = answer_info.get("timestamp", None)
                    db_answer.text = answer_info["text"]
                    db_answer.user_did_not_find_answer = answer_info.get("user_did_not_find_answer")

                    self.db_session.merge(db_answer)

            self.db_session.merge(db_assignment)
            self.db_session.commit()


def generate_template(timeline, questions):
    return question_template.render(
        timeline_entries=sorted(timeline.dates_to_summaries.items()),
        questions=questions,
        len=len,
        format_text=format_text)


def read_questions(f_db, f_anno):
    db_lines = iter(list(f_db))
    anno_lines = iter([line for line in f_anno if not line.startswith("#")])

    tl_questions = defaultdict(list)
    while True:
        db_line = next(db_lines, None)
        if db_line is None:
            break
        anno_line = next(anno_lines)
        tl_topic, tl_name, question_type, info = db_line.strip().split("\t", 4)

        anno_line = anno_line.strip()

        if len(anno_line) == 0:
            continue

        if anno_line[0] == "-":
            continue

        if question_type == "detail":
            answer_line = next(anno_lines)
            assert answer_line.startswith(">>")
            answer = answer_line[2:]
        else:
            answer = info

        tl_questions[tl_topic, tl_name].append(Question(anno_line, question_type, answer))

    return tl_questions


def format_text(text):
    doc = nlp(text)
    better_sents = []
    for sent in doc.sents:
        better_sent = sent.text.capitalize()
        better_sent = better_sent.replace(" .", ".").replace(" 's", "'s").replace(" , ", ", ").replace("-lrb- ", "(").replace(" -rrb-", ")").replace(" !", "!").replace(" ?", "?").strip()
        better_sents.append(better_sent)

    return " ".join(better_sents)


def generate_question_html(sys_tl_dir, tl_questions):
    for (tl_topic, tl_name), questions in tl_questions.items():
        sys_tl_path = os.path.join(sys_tl_dir, tl_topic, tl_name)

        with open(sys_tl_path, errors="ignore") as f_tl:
            tl = Timeline.from_file(f_tl)

        yield tl_topic, tl_name, generate_template(tl, questions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sys_tl_dir")
    parser.add_argument("questions_db")
    parser.add_argument("questions_annotator_file")

    args = parser.parse_args()

    with open(args.questions_db) as f_db, open(args.questions_annotator_file) as f_anno:
        tl_questions = read_questions(f_db, f_anno)

    if True:
        MTurkClient().upload_batch({
            "html": html,
            "questions": tl_questions[tl_topic, tl_name],
            "topic": tl_topic,
            "system": os.path.basename(args.sys_tl_dir),
            "timeline_name": tl_name} for tl_topic, tl_name, html in generate_question_html(args.sys_tl_dir, tl_questions)
        )
    else:
        for _, _, q in generate_question_html(args.sys_tl_dir, tl_questions):
            print(q)
            break


if __name__ == "__main__":
    main()
