


from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
import numpy as np
import time 

def euclidean_distance(u, v):
    return np.sqrt(sum([(a - b) ** 2 for a, b in zip(u, v)]))


#(a) runMapReduce(pointsRDD,k,L): implements the 4-approximation 
# MapReduce algorithm for diversity maximization described above. More specifically, 
# it receives in input an RDD of points (pointsRDD), and two integers, k and L, 
# and performs the following activities.

def runMapReduce(pointsRDD,k,L):
    return 0

#(b) measure(pointsSet): receives in input a set of points (pointSet) and computes 
# the average distance between all pairs of points. The set pointSet must be 
# represented as ArrayList<Vector> (in Java) or list of tuple (in Python).

def measure(pointsSet):
    return 0


def main():

    assert len(sys.argv) == 3, "Two arguments must be provided: k desired number of centers, S set of points"
    assert str(sys.argv[1]).isdigit(), "k must be a number"
    
    inputPath = sys.argv[2]
    L = sys.argv[1]
    # SPARK SETUP
    conf = SparkConf().setAppName('HW3') #.setMaster("local[*]") not needed
    sc = SparkContext(conf=conf)

    #to read the input
    sc.textFile(inputPath).map(lambda x:x).repartition(L).cache()



if __name__ == "__main__":
    main()
