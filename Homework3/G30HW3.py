#this part needed just for me Leo
#import findspark
#findspark.init('/home/leokane96/SparkDir/spark-3.0.0-preview2-bin-hadoop2.7')
#from pyspark import SparkContext, SparkConf

import sys
import os
import random as rand
import numpy as np
import time 

from runSequential import runSequential

#TODO this global variable is needed to specify first parameter kCenterMPD when used as Higher order function, find better solution
global K 
K=-1

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

    # TOT: O(k * (|S| + |S|)) = O(k*|S|)

    return list(S)
    # else:
    #     return 0

def callKcenter(S):
    global K 
    return kCenterMPD(S, K) 


#(a) runMapReduce(pointsRDD,k,L): implements the 4-approximation 
# MapReduce algorithm for diversity maximization described above. More specifically, 
# it receives in input an RDD of points (pointsRDD), and two integers, k and L, 
# and performs the following activities.

def runMapReduce(pointsRDD,k,L):
    #useless can be removed, (usefull to understand what K is)
    global K
    K = k 
    #######################
    #get coreset using kcenterMPD
    coreset = pointsRDD.repartition(L).mapPartitions(callKcenter).collect()
    #return k point obtained from runsequential
    return runSequential(coreset, K)

#(b) measure(pointsSet): receives in input a set of points (pointSet) and computes 
# the average distance between all pairs of points. The set pointSet must be 
# represented as ArrayList<Vector> (in Java) or list of tuple (in Python).

def measure(pointsSet):
    return 0


def main():
    global K

    assert len(sys.argv) == 4, "Two arguments must be provided: k desired number of centers, S set of points"
    assert str(sys.argv[1]).isdigit(), "k must be a number"
    assert str(sys.argv[3]).isdigit(), "L = number of partitions"

    L = int(sys.argv[3])
    inputPath = sys.argv[2]
    K = int(sys.argv[1])
    # SPARK SETUP
    conf = SparkConf().setAppName('HW3') #.setMaster("local[*]") not needed
    sc = SparkContext(conf=conf)

    #to read the input and convert tuple of string to tuple of float
    docs = sc.textFile(inputPath).map(lambda x: tuple(float(dim) for dim in x.split(",")))

    print(runMapReduce(docs, K, L))

if __name__ == "__main__":
    main()
