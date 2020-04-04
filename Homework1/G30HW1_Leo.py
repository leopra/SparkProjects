import findspark

import operator 

findspark.init('/home/leokane96/SparkDir/spark-3.0.0-preview2-bin-hadoop2.7')

from termcolor import colored

from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand


def class_count_1(docs, K):
    def map1(document):
        pair_split = document.split(' ')
        return [(int(pair_split[0]) % K, pair_split[1])]

    def gather_pairs(pairs):
        pairs_dict = {}
        for p in pairs[1]:
            if p not in pairs_dict.keys():
                pairs_dict[p] = 1
            else:
                pairs_dict[p] += 1

        return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

    word_count = (docs.flatMap(map1)  # <-- MAP PHASE (R1)
                  .groupByKey()  # <-- REDUCE PHASE (R1)
                  .flatMap(gather_pairs)
                  .reduceByKey(lambda x, y: x + y))  # <-- REDUCE PHASE (R2)
    return word_count


def class_count_2_with_partition(docs):

    def map1(document):
        pair_split = document.split(' ')
        return [(int(pair_split[0]), pair_split[1])]

    def gather_pairs_partitions(pairs):
        pairs_dict = {}
        count=0
        maxpartition="MaxPartition"
        for p in pairs:
            # swapping key and value
            key, aclass = p[0], p[1]
            if aclass not in pairs_dict.keys():
                pairs_dict[aclass] = 1
            else:
                pairs_dict[aclass] += 1
            count += 1
        pairs['MaxPartion'] = count
        print(type(aclass))
        print(type(pairs_dict[aclass]))
        return [(aclass, pairs_dict[aclass]) for aclass in pairs_dict.keys()]

    word_count = (docs.flatMap(map1)  # <-- MAP PHASE (R1)
                  .mapPartitions(gather_pairs_partitions)  # <-- REDUCE PHASE (R1)
                  .groupByKey()  # <-- REDUCE PHASE (R2)
                  .mapValues(lambda vals: sum(vals)))
    return word_count


def main():

    assert len(sys.argv) == 3, "Usage: python TemplateHW1.py <K> <file_name>"

    
    # SPARK SETUP
    conf = SparkConf().setAppName('TemplateHW1').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # INPUT READING

    # Read number of partitions
    K = sys.argv[1]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    # Read input file and subdivide it into K random partitions
    data_path = sys.argv[2]
    assert os.path.isfile(data_path), "File or folder not found"
    docs = sc.textFile(data_path, minPartitions=K).cache()
    docs.repartition(numPartitions=K)

    numdocs = docs.count()
    print("Number of documents = ", numdocs)


    #print("CLASS COUNT 1 = ", class_count_1(docs, K).collect())

    
    stats = class_count_2_with_partition(docs)
    maxvalue= stats.max(key=lambda x:x[1]) 
    print("MOST FREQUENT CLASS = ", maxvalue)
    #TODO ties must be broken in favor of the smaller class in alphabetical order

    print("Max partition size = ", )

if __name__ == "__main__":
    main()

