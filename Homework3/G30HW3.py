from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
import numpy as np
import time 
from runSequential import runSequential

def euclidean_distance(u, v):
    return np.sqrt(sum([(a - b) ** 2 for a, b in zip(u, v)]))

def kCenterMPD(S, k):
    S = list(S)
    assert k < len(S), "k must be less than the size of S"

    P = set(S)
    S = set()

    p = P.pop()
    S.add(p)

    # dists_dict is a support data structure to keep a reference of the (a dict, in this implementation)
    # the first initialization saves the distance of each point from the first selected point p.
    # in the following for loop, the dict is updated once for every new center added to the resulting set,
    # keeping the minimum between the current distance and the computed distance from the new center.
    # This should ensure that the entire method has a complexity of O(k*|S|)

    dists_dict = {key: euclidean_distance(key, p) for key in P}

    for i in range(1, k):  # O(k)
        c = max(dists_dict, key=dists_dict.get)  # O(|S|)
        for t in P:  # O(|S|)
            dists_dict[t] = min(dists_dict[t], euclidean_distance(t, c))
        S.add(c)
        P.discard(c)

    return list(S)

def callKcenter(S):
    global K
    return kCenterMPD(S, K)

#(a) runMapReduce(pointsRDD,k,L): implements the 4-approximation 
# MapReduce algorithm for diversity maximization described above. More specifically, 
# it receives in input an RDD of points (pointsRDD), and two integers, k and L, 
# and performs the following activities.

def runMapReduce(pointsRDD,k,L):
    global K
    K = k

    start = time.time()
    temp = pointsRDD.mapPartitions(callKcenter).cache()
    r1_elapsed = time.time() - start

    #get coreset using kcenterMPD
    start = time.time()
    coreset = temp.collect()
    #return k point obtained from runsequential
    solution = runSequential(coreset, K)
    r2_elapsed = time.time() - start

    return solution, r1_elapsed, r2_elapsed

#(b) measure(pointsSet): receives in input a set of points (pointSet) and computes 
# the average distance between all pairs of points. The set pointSet must be 
# represented as ArrayList<Vector> (in Java) or list of tuple (in Python).

def measure(pointSet):

    tot = 0.
    for i in range(len(pointSet) - 1):
        for j in range(i + 1, len(pointSet)):
            tot += euclidean_distance(pointSet[i], pointSet[j])

    return tot / ((len(pointSet) * (len(pointSet) - 1)) / 2)

K = 0


def main():
    assert len(sys.argv) == 4, "Two arguments must be provided: k desired number of centers, S set of points"
    assert str(sys.argv[2]).isdigit(), "k must be a number"
    assert str(sys.argv[3]).isdigit(), "L must be a number"

    L = int(sys.argv[3])
    inputPath = sys.argv[1]
    k = int(sys.argv[2])
    # SPARK SETUP
    conf = SparkConf().setAppName('HW3')
    sc = SparkContext(conf=conf)

    # to read the input and convert tuple of string to tuple of float
    start = time.time()
    docs = sc.textFile(inputPath).map(lambda x: tuple(float(dim) for dim in x.split(","))).repartition(L).cache()
    init_time = time.time() - start

    print("Number of points =" , docs.count())
    print("k = ", k)
    print("L = ", L)
    print("Initialization time = ", init_time * 1000)

    solution, r1_time, r2_time = runMapReduce(docs, k, L)

    print("Runtime of Round 1 = ", r1_time * 1000)
    print("Runtime of Round 2 = ", r2_time * 1000)

    avg = measure(solution)
    print("Average distance = ", avg)

if __name__ == "__main__":
    main()