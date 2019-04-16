import pickle
import sys
from collections import Counter
import glob

import matplotlib.pyplot as plt


def compute_size_stats(fnames):
    size_stats = []
    all_cl_sizes = set()
    for cluster_file in fnames:
        with open(cluster_file, "rb") as f:
            clusters = pickle.load(f)

            clusters_size_stats = Counter(map(len, clusters))
            size_stats.append(clusters_size_stats)

            all_cl_sizes.update(clusters_size_stats.keys())

    return size_stats, all_cl_sizes


if __name__ == "__main__":
    size_stats = []
    all_cl_sizes = set()
    for pattern in sys.argv[1:]:
        pat_size_stats, pat_all_cl_sizes = compute_size_stats(glob.glob(pattern))
        size_stats.append(sum(pat_size_stats, Counter()))
        all_cl_sizes.update(pat_all_cl_sizes)

    for cl_size in all_cl_sizes:
        print(cl_size, "\t".join([str(s.get(cl_size)) for s in size_stats]))

    _, axes = plt.subplots(1, len(size_stats), sharey=True, sharex=True)

    for stats, plot in zip(size_stats, axes):
        plot.bar(*zip(*sorted(stats.items())))

    plt.show()
