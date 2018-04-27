from tilse.data.timelines import Timeline

import sys

if __name__ == "__main__":
    timelines = []
    for tl_fname in sys.argv[1:]:
        with open(tl_fname, encoding="latin-1") as f:
            #print(tl_fname)
            timelines.append(Timeline.from_file(f))

    timeline_dates = [sorted(tl.get_dates()) for tl in timelines]

    rows = []
    for date_idx in range(5):
        date_row = []
        sum_row = []
        for tl_idx in range(len(timelines)):
            tl_date = timeline_dates[tl_idx][date_idx]
            tl_sum = timelines[tl_idx][tl_date]
            date_row.append("\\textbf{" + str(tl_date) + "}")
            sum_row.append(str(" \\newline ".join(tl_sum)))

        rows.append(date_row)
        rows.append(sum_row)

    for row in rows:
        print(" & ".join(row), "\\\\\\hline")
