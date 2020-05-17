


from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
import TupleInput as ti
import numpy as np
import time 

def euclidean_distance(u, v):
    return np.sqrt(sum([(a - b) ** 2 for a, b in zip(u, v)]))

#receives in input a set of points S and returns the max distance between two 
# points in S.
def exactMPD(S):
    max_dist = 0

    for i in range(0, len(S) - 1):
        for j in range(i + 1, len(S)):
            max_dist = max(euclidean_distance(S[i], S[j]), max_dist)

    return max_dist

#receives in input a set of points S and an interger k < |S|, selects k points 
# at random from S (let S' denote the set of these k points) and returns the 
# maximum distance d(x,y), over all x in S' and y in S. 
# Define a constant SEED in your main program 

def twoApproxMPD(S, k):
    assert k < len(S), "k must be less than the size of S"

    co_1 = []
    max_ds = 0
    k_points = rand.sample(range(0, len(S)), k)

    for i in k_points:
        co_1.append(S[i])

    for co in S:
        for point in co_1:
            max_ds = max(euclidean_distance(co, point), max_ds)

    return max_ds

 

#receives in input a set of points S and an integer k < |S|, and returns a set C 
# of k centers selected from S using the Farthest-First Traversal algorithm. 
# It is important that kCenterMPD(S,k) run in O(|S|*k) time
def kCenterMPD(S, k):
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

    # TOT: O(k * (|S| + |S|)) = O(k*|S|)

    return list(S)


def main():
    rand.seed(42)

    assert len(sys.argv) == 3, "Two arguments must be provided: k desired number of centers, S set of points"
    assert str(sys.argv[1]).isdigit(), "k must be a number"

    # SPARK SETUP
    #conf = SparkConf().setAppName('HW2').setMaster("local[*]")
    #sc = SparkContext(conf=conf)

    inputPoints = ti.readTuplesSeq(sys.argv[2])
    k = int(sys.argv[1])

    start = time.time()
    result = exactMPD(inputPoints)
    elapsed = time.time() - start
    print('EXACT ALGORITHM')
    print(f'Max distance = {result}')
    print(f'Running time = {elapsed * 1000}')

    print('')

    start = time.time()
    result = twoApproxMPD(inputPoints, k)
    elapsed = time.time() - start

    print('2-APPROXIMATION ALGORITHM')
    print(f'k = {k}')
    print(f'Max distance = {result}')
    print(f'Running time = {elapsed * 1000}')

    print('')

    start = time.time()
    centers = kCenterMPD(inputPoints, k)
    result = exactMPD(centers)
    elapsed = time.time() - start

    print('k-CENTER-BASED ALGORITHM')
    print(f'k = {k}')
    print(f'Max distance = {result}')
    print(f'Running time = {elapsed * 1000}')


if __name__ == "__main__":
    main()
