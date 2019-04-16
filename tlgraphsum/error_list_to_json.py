import sys
import re
from collections import defaultdict
import json


def main():
    per_corpus_problems = defaultdict(lambda: defaultdict(set))
    per_corpus_topic_problems = defaultdict(set)
    with open(sys.argv[1]) as f:
        for line in f:
            match = re.search("(crisis.data|timeline17)/(.*?)/.*/(.*?)\.", line)

            if not match:
                print("Ignoring", line)
                continue

            corpus_name = "timeline17" if match.group(1) == "timeline17" else "crisis"
            topic_name = match.group(2)
            doc_id = match.group(3)

            per_corpus_problems[corpus_name][topic_name].add(doc_id)
            corpus_id = "tl17" if corpus_name == "timeline17" else "crisis"
            per_corpus_topic_problems[corpus_id + "-" + topic_name].add(doc_id)

    per_corpus_problems_as_list = defaultdict(dict)

    for corpus_name, errors in per_corpus_problems.items():
        for topic_name, fids in errors.items():
            per_corpus_problems_as_list[corpus_name][topic_name] = list(fids)

    print(dict(per_corpus_topic_problems.items()))

    with open("problematic_files.json", "w") as f:
        json.dump(per_corpus_problems_as_list, f)


if __name__ == "__main__":
    main()
