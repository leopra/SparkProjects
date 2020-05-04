import findspark

findspark.init('/home/leokane96/SparkDir/spark-3.0.0-preview2-bin-hadoop2.7')

from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
import TupleInput as ti
import math 
import numpy as np

rand.seed(42)

#receives in input a set of points S and returns the max distance between two 
# points in S.

def exactMPD(S, K):
    return 0


#receives in input a set of points S and an interger k < |S|, selects k points 
# at random from S (let S' denote the set of these k points) and returns the 
# maximum distance d(x,y), over all x in S' and y in S. 
# Define a constant SEED in your main program 

def twoApproxMPD(S,k):

    #copy the points between all clusters
    L = S.count()-1
    kpoints = rand.sample(range(1,L), k)
    K=k

    def copyKPoints(coord):
        mappedpairs=[]
        if coord[0] in kpoints:
            for i in range(k):
                mappedpairs.append((i, (0, coord[1])))

        else:
            mappedpairs.append((coord[0] % k, (coord[0], coord[1])))

        return mappedpairs


    #return the maxpairwise of each clusters    
    def getMax(coords):
        coords = list(coords)
        print('numberofpoints: ' , len(coords[0][1]))
        maxds = 0

        co1 = []
        
        coords = list(coords[0][1]) #TODO for some reason coords comes packed into an array, here i unwrap it
        
        if (len(coords)>K):
            for i in range(K):
                co1.append(coords[i])

        for co in coords:
            #print(co)
            a = np.array(co[1])
            for point in co1:
                b = np.array(point[1])
                temp = np.linalg.norm(a-b)
            if temp > maxds:
                maxds = temp
        
        k = [(0,maxds)]
        return k


    g = (S.flatMap(copyKPoints)     #using a single worker, get k points and copy it, one for each cluster (as many clusters as points)
          .repartition(K)           #now we add multiple workers
          .groupByKey()             #group by key
          .mapValues(sorted)        #sort so the S' points are first first element of tuple is 0
          .mapPartitions(getMax)    #apply the getMax alg to each worker
          .reduceByKey(max).values()) #get the max of all results

    return g  


#receives in input a set of points S and an integer k < |S|, and returns a set C 
# of k centers selected from S using the Farthest-First Traversal algorithm. 
# It is important that kCenterMPD(S,k) run in O(|S|*k) time
def kCenterMPD(S,k):
    return 0


def main():
    rand.seed(42)

    assert len(sys.argv) == 3
    K = int(sys.argv[1])

    # SPARK SETUP
    conf = SparkConf().setAppName('HW2').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    document = sys.argv[2] 
    #there should be a better way to use the function given by the professor ?
    rdd = sc.textFile(document, minPartitions=1).map(lambda i : tuple(float(dim) for dim in i.split(','))).zipWithIndex()
    rdd = rdd.map(lambda x: (x[1], x[0]))    
    rdd = twoApproxMPD(rdd, K)
    print("MAX PAIRWISE DISTANCE: ", rdd.collect())
    print("npartitions: ", rdd.getNumPartitions())

   

if __name__ == "__main__":
    main()
