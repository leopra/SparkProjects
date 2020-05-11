


from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
import TupleInput as ti
import numpy as np
import time 

#receives in input a set of points S and returns the max distance between two 
# points in S.
def exactMPD(S):
    return 0

#receives in input a set of points S and an interger k < |S|, selects k points 
# at random from S (let S' denote the set of these k points) and returns the 
# maximum distance d(x,y), over all x in S' and y in S. 
# Define a constant SEED in your main program 

def twoApproxMPD(S,k):

    assert k < len(S)
    co1=[]
    maxds=0
    L = len(S)-1
    kpoints = rand.sample(range(1,L), k)

    for i in kpoints:
        co1.append(S[i])

    for co in S:
        for point in co1:
            temp = np.sqrt(sum([(x-z)**2 for x, z in zip(co,point)]))
        if temp > maxds:
            maxds = temp

    return maxds

 

#receives in input a set of points S and an integer k < |S|, and returns a set C 
# of k centers selected from S using the Farthest-First Traversal algorithm. 
# It is important that kCenterMPD(S,k) run in O(|S|*k) time
def kCenterMPD(S,k):
    return 0


def main():
    rand.seed(42)

    assert len(sys.argv) == 3

    # SPARK SETUP
    #conf = SparkConf().setAppName('HW2').setMaster("local[*]")
    #sc = SparkContext(conf=conf)

    data = ti.readTuplesSeq(sys.argv[2])
    s = time.time()
    print('twoApproxMPD: ', twoApproxMPD(data, int(sys.argv[1])))
    e = time.time()
    print('time taken: ', e-s)
if __name__ == "__main__":
    main()
