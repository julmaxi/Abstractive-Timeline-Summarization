import sys
import os


def run_on_files(long_filename, short_filename):
    with open(long_filename) as f_long, open(short_filename) as f_short:
        long_linenum = 0
        num_changes = 0
        num_changes_included = 0

        tl1_len = 0
        tl2_len = 0

        while True:
            try:
                sys_1_sent, sys_1_date, sys_1_included = next(f_long).split("\t")
                long_linenum += 1
                sys_2_sent, sys_2_date, sys_2_included = next(f_short).split("\t")

                sys_1_sent = sys_1_sent.strip()
                sys_2_sent = sys_2_sent.strip()

                while sys_1_sent != sys_2_sent:
                    sys_1_sent, sys_1_date, sys_1_included = next(f_long).split("\t")
                    long_linenum += 1
                    sys_1_sent = sys_1_sent.strip()

                if sys_1_date != sys_2_date:
                    num_changes += 1
                    if sys_1_included.strip() == "True" or sys_2_included.strip() == "True":
                        num_changes_included += 1

                if sys_1_included.strip() == "True":
                    tl1_len += 1
                elif sys_2_included.strip() == "True":
                    tl2_len += 1

            except StopIteration:
                break

        return num_changes, num_changes_included, long_linenum, tl1_len, tl2_len


def main():
    extractive_files = []
    ap_extractive_files = []

    for filename in os.listdir("."):
        if not filename.startswith("candidate-debug-info"):
            continue

        if "repetition" in filename:
            continue

        if "ap-extractive-oracle" in filename:
            ap_extractive_files.append(filename)
        elif "extractive-oracle" in filename:
            extractive_files.append(filename)

    print(ap_extractive_files)
    print(len(extractive_files), len(ap_extractive_files))

    assert len(extractive_files) == len(ap_extractive_files)

    for ex_filename, ap_filename in zip(sorted(extractive_files), sorted(ap_extractive_files)):
        print(ex_filename, ap_filename)
        num_changes, num_changes_included, long_linenum, tl1_len, tl2_len = run_on_files(ex_filename, ap_filename)
        print(num_changes, num_changes_included, max(tl1_len, tl2_len))



if __name__ == "__main__":
    main()
