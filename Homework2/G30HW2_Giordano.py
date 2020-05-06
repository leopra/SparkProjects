from pyspark import SparkContext, SparkConf
import sys
import os
import numpy as np
import random as rand
from Homework2 import TupleInput as ti

def euclidean_distance(u, v):
    return np.sqrt(sum([(a - b) ** 2 for a, b in zip(u, v)]))

#receives in input a set of points S and an integer k < |S|, and returns a set C 
# of k centers selected from S using the Farthest-First Traversal algorithm. 
# It is important that kCenterMPD(S,k) run in O(|S|*k) time
def kCenterMPD(S,k):
    assert k < len(S), "k must be less than the size of S"

    P = set(S)
    S = set()

    p = P.pop()
    S.add(p)
    dists_dict = {key: euclidean_distance(key, p) for key in P}
    for i in range(1, k):
        c = max(dists_dict, key=dists_dict.get)
        print(f'{c}')
        for t in P:
            dists_dict[t] = min(dists_dict[t], euclidean_distance(t, c))
        S.add(c)
        P.discard(c)

    return list(S)


def main():
    rand.seed(42)

    assert len(sys.argv) == 3, "Two arguments must be provided: k desired number of centers, S set of points"
    assert str(sys.argv[1]).isdigit(), "k must be a number"

    # SPARK SETUP
    conf = SparkConf().setAppName('HW2').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    data = ti.readTuplesSeq(sys.argv[2])
    print(data[:10])

    k_centers = kCenterMPD(data, int(sys.argv[1]))
    print(f'{k_centers}')

if __name__ == "__main__":
    main()
