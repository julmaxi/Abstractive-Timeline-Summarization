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

import subprocess
import argparse
import datetime

from pathlib import PurePath, Path
import json


SANDBOX_ENDPOINT = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"


with open(os.path.join(os.path.dirname(__file__), "bwscalingtemplate.html")) as file_:
    scaling_template = Template(file_.read())


def str_to_tl_date(s):
    return datetime.date(*map(int, s.split("-")))


def generate_template(timeline, candidate_tuples, event_info):
    return scaling_template.render(
        timeline_entries=sorted(timeline.dates_to_summaries.items()),
        candidate_tuples=candidate_tuples,
        event_info=event_info,
        len=len)

class MTurkClient:
    QUESTION = """
<HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
  <HTMLContent><![CDATA[
  {html}
  ]]></HTMLContent>
  <FrameHeight>0</FrameHeight>
</HTMLQuestion>
"""

    def __init__(self, task_name, sandbox=True):
        args = {}
        if sandbox:
            args["endpoint_url"] = SANDBOX_ENDPOINT
        self.client = boto3.client('mturk', **args)
        self.sandbox = sandbox

        self._hittype = None
        self._tester_qualification = None

        self.task_name = task_name

    def list_reviewable_hit_ids_and_annotations(self):
        return self.fetch_multipage_operation(
            self.client.list_hits,
            {},
            collector=lambda r: [(h["HITId"], h["RequesterAnnotation"]) for h in r["HITs"] if h["HITTypeId"] == self.hittype]
        )

        return self.fetch_multipage_operation(
            self.client.list_reviewable_hits,
            {"HITTypeId": self.hittype},
            collector=lambda r: [(h["HITId"], h["RequesterAnnotation"]) for h in r["HITs"]]
        )

    def parse_answer(self, answer_xml):
        ns = {'mturk': "http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/QuestionFormAnswers.xsd"}
        tree = ET.fromstring(answer_xml)
        answers = defaultdict(dict)

        for answer in tree.findall(".//mturk:Answer", ns):
            question_id = answer.find("mturk:QuestionIdentifier", ns).text
            answer_text = answer.find("mturk:FreeText", ns).text

            answers[question_id] = answer_text

        return answers

    def load_answers_for_hit(self, hit_id):
        return self.fetch_multipage_operation(
            self.client.list_assignments_for_hit,
            {"HITId": hit_id},
            collector=lambda r: [[
                (
                    a["WorkerId"],
                    self.parse_answer(a["Answer"])
                ) for a in r["Assignments"]]
            ]
        )

    def create_hittype(self):
        qualifications = []
        if self.sandbox:
            qualifications.append({"QualificationTypeId": self.tester_qualification, "RequiredToPreview": True, "Comparator": "Exists"})

        response = self.client.create_hit_type(
            AutoApprovalDelayInSeconds=60 * 60 * 24,
            AssignmentDurationInSeconds=60 * 60 * 5,
            Reward='0.01',
            Title=self.task_name,
            Keywords='nlp',
            Description='Test',
            QualificationRequirements=qualifications
        )

        return response["HITTypeId"]

    @property
    def tester_qualification(self):
        if self._tester_qualification:
            return self._tester_qualification

        qualification_types = self.fetch_multipage_operation(
            self.client.list_qualification_types, {"MustBeRequestable":True, "MustBeOwnedByCaller": True}, collector=lambda i: i["QualificationTypes"])

        for qualification_type in qualification_types:
            if qualification_type['Name'] == "Test HITs":
                self._tester_qualification = qualification_type["QualificationTypeId"]
                return self._tester_qualification

        response = self.client.create_qualification_type(
            Name="Test HITs",
            QualificationTypeStatus="Active",
            Description="Allows testing hits")
        self._tester_qualification = response["QualificationType"]["QualificationTypeId"]
        return self._tester_qualification

    @property
    def hittype(self):
        if self._hittype:
            return self._hittype

        hit_type_id = self.create_hittype()
        self._hittype = hit_type_id
        return hit_type_id

    def upload_html(self, pages):
        hit_ids = []
        for identifier, page in pages:
            response = self.client.create_hit_with_hit_type(
                HITTypeId=self.hittype,
                MaxAssignments=1,
                LifetimeInSeconds=60 * 60 * 24 * 14,
                Question=self.QUESTION.format(html=page),
                RequesterAnnotation=identifier)

            hit_ids.append(response["HIT"]["HITId"])

        return hit_ids

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


def generate_date_tuples(tl):
    generate_bw_proc = subprocess.run(["perl", os.path.join(os.path.dirname(__file__), "generate-BWS-tuples.pl")], input="\n".join(str(d) for d in tl.get_dates()).encode("utf-8"), stdout=subprocess.PIPE)

    result_tuples = generate_bw_proc.stdout.decode("utf-8").strip().split("\n")

    return result_tuples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("timeline")
    parser.add_argument("event_info")

    parser.add_argument("--tuple-file")

    args = parser.parse_args()

    tl_path = PurePath(args.timeline)
    tl_topic = tl_path.parts[-2]
    tl_name = tl_path.parts[-1].split(".")[0]

    with open(args.event_info) as f:
        all_event_info = json.load(f)

    event_info = all_event_info[tl_topic]

    with open(args.timeline, errors="ignore") as f:
        tl = Timeline.from_file(f)

    if args.tuple_file is None:
        result_tuples = generate_date_tuples(tl)
    else:
        with open(args.tuple_file) as f:
            result_tuples = f.read().strip().split("\n")[1:]

    tuples = [
        tuple(sorted((str_to_tl_date(date), tl.dates_to_summaries[str_to_tl_date(date)]) for date in tuple_line.split("\t"))) for tuple_line in result_tuples
    ]

    batches = []

    curr_idx = 0

    while curr_idx < len(tuples):
        batches.append(tuples[curr_idx:curr_idx + 20])
        curr_idx += 20

    print("Uploading", len(tuples), "tuples")

    client = MTurkClient("Timeline Importance Annotation VIII (K)")

    hit_info_path = Path("./hitinfo")
    hit_info_path.mkdir(parents=True, exist_ok=True)

    for batch in batches:
        hit_ids = client.upload_html([("importance:" + tl_topic + "__" + tl_name, generate_template(tl, batch, event_info))])

        hit_id = hit_ids[0]

        hit_info_file_path = hit_info_path / hit_id
        with open(hit_info_file_path, "w") as f:
            f.write(tl_topic)
            f.write("\t")
            f.write(tl_name)
            f.write("\n")
            for candidate_tuple in batch:
                f.write("\t".join(map(lambda it: str(it[0]), candidate_tuple)))
                f.write("\n")


if __name__ == "__main__":
    main()
