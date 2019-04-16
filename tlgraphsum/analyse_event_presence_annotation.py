import sys
from collections import defaultdict
from tilse.data.timelines import Timeline
import os
import datetime


def main():
    event_annotations = defaultdict(dict)
    current_tl_key = None

    system_name = None

    if len(sys.argv) == 3:
        system_name = sys.argv[2]

    with open(sys.argv[1]) as f:
        for lidx, line in enumerate(f):
            if line.startswith("=="):
                _, corpus, tl_name = line.split()
                current_tl_key = (corpus, tl_name)
                continue

            elems = line.split()
            if len(elems) == 1:
                continue

            date, response = elems

            year, month, day = map(int, date.split("-"))

            event_annotations[current_tl_key][datetime.date(year, month, day)] = response

    event_present_in_first_3_ratios = {}
    event_present_in_last_3_ratios = {}
    event_present_ratios = {}
    for tl_key, annotations in event_annotations.items():
        dates_to_consider = None
        if system_name:
            with open(os.path.join("gold-timelines", tl_key[0], tl_key[1]), errors="ignore") as f:
                gold_tl = Timeline.from_file(f)

            with open(os.path.join("system_timelines", system_name, tl_key[0], tl_key[1]), errors="ignore") as f:
                sys_tl = Timeline.from_file(f)

            dates_to_consider = set(gold_tl.get_dates()).intersection(set(sys_tl.get_dates()))

        num_correct_first_3 = 0
        num_correct_last_3 = 0
        num_correct = 0
        num_total = 0
        annotations = annotations.items()

        for idx, (date, annotation) in enumerate(sorted(annotations)):
            if dates_to_consider and date not in dates_to_consider:
                continue
            if annotation != "-":
                #print("==>", annotation)
                num_correct += 1

                if idx < 3:
                    num_correct_first_3 += 1
                elif idx >= (len(annotations) - 3):
                    num_correct_last_3 += 1
            num_total += 1

        if num_total == 0:
            event_present_ratios[tl_key] = 0
        else:
            event_present_ratios[tl_key] = num_correct / num_total
        event_present_in_first_3_ratios[tl_key] = num_correct_first_3 / 3
        event_present_in_last_3_ratios[tl_key] = num_correct_last_3 / 3

    #print(event_present_ratios)

    print("Last", sum(event_present_in_last_3_ratios.values()) / len(event_present_in_last_3_ratios))
    print("First", sum(event_present_in_first_3_ratios.values()) / len(event_present_in_first_3_ratios))
    print("Total", sum(event_present_ratios.values()) / len(event_present_ratios))


if __name__ == "__main__":
    main()
