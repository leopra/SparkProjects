from pyspark import SparkContext, SparkConf
import sys
import os
import numpy as np
import random as rand
import TupleInput as ti
import time

start_time = time.time()
def distance(m, n):
    return np.sqrt((m[0] - n[0]) ** 2 + (m[1] - n[1]) ** 2)


def exactMPD(S):
    max_dist = distance(S[0], S[1])
    for i in range(0, len(S) - 1):
        for j in range(i + 1, len(S)):
            max_dist = max(distance(S[i], S[j]), max_dist)

    return max_dist

def main():
    rand.seed(42)

    assert len(sys.argv) == 3, "Two arguments must be provided: k desired number of centers, S set of points"
    assert str(sys.argv[1]).isdigit(), "k must be a number"

    # SPARK SETUP
    conf = SparkConf().setAppName('HW2').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    data = ti.readTuplesSeq(sys.argv[2])

    exact = exactMPD(data)
    print('Max distance = ' + str(exact))
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
