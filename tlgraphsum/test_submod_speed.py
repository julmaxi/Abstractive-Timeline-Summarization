from submodular import SubModularOptimizer, RedundancyFactor, ClusterMembershipConstraint, MaxDateCountConstraint, KnapsackConstraint, SubsetKnapsackConstraint

import random

from collections import defaultdict

import logging

logging.basicConfig(level=logging.DEBUG)

def random_sentence():
    WORDS = ["this", "that", "man", "woman", "dwarf", "elefant"]
    return [random.choice(WORDS)]


def test_submod(n=500000):
    constraints = []
    factors = []

    constraints.append(MaxDateCountConstraint(2, dict(enumerate([i for i in range(n)]))))
    constraints.append(KnapsackConstraint(5, dict((i, 1) for i in range(n))))
    constraints.append(ClusterMembershipConstraint(dict(enumerate([random.randint(0, 100) for _ in range(n)]))))
    constraints.append(SubsetKnapsackConstraint(1, dict((i, 1) for i in range(n)), {1, 2, 3, 4}))

    red_factor = RedundancyFactor(
        dict(enumerate([random.random() for _ in range(n)])),
        dict(enumerate([random.randint(0, 100) for _ in range(n)]))
    )

    factors.append(red_factor)

    opt = SubModularOptimizer(
        factors,
        constraints
    )

    result = opt.run(list(range(n)))

    print(result)


if __name__ == "__main__":
    test_submod()
