from tilse.data.timelines import Timeline
import sys
import os


def main():
    with open(sys.argv[2], "w") as f_out:
        for topic_dirname in os.listdir(sys.argv[1]):
            for tl_fname in os.listdir(os.path.join(sys.argv[1], topic_dirname)):
                with open(os.path.join(sys.argv[1], topic_dirname, tl_fname), errors="ignore") as f_tl:
                    tl = Timeline.from_file(f_tl)

                f_out.write("== {} {}\n".format(topic_dirname, tl_fname))
                for date in sorted(tl.get_dates()):
                    f_out.write(str(date))
                    f_out.write("\t\t\n")


if __name__ == "__main__":
    main()
