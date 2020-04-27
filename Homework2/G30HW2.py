


from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
import TupleInput as ti

#receives in input a set of points S and returns the max distance between two 
# points in S.
def exactMPD(S):
    return 0

#receives in input a set of points S and an interger k < |S|, selects k points 
# at random from S (let S' denote the set of these k points) and returns the 
# maximum distance d(x,y), over all x in S' and y in S. 
# Define a constant SEED in your main program 
def twoApproxMPD(S,k):
    return 0

#receives in input a set of points S and an integer k < |S|, and returns a set C 
# of k centers selected from S using the Farthest-First Traversal algorithm. 
# It is important that kCenterMPD(S,k) run in O(|S|*k) time
def kCenterMPD(S,k):
    return 0


def main():
    rand.seed(42)

    assert len(sys.argv) == 3

    # SPARK SETUP
    conf = SparkConf().setAppName('HW2').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    data = ti.readTuplesSeq(sys.argv[2])
    print(data[:10])

if __name__ == "__main__":
    main()
