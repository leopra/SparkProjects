from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand

def class_count(pairs, K):

    def map_pair(pair):
        pair_split = pair.split(' ', 1)
        return [(int(pair_split[0]) % K, pair_split[1])]

    def gather_pairs(pairs):
        pairs_dict = {}
        for c in pairs[1]:
            if c not in pairs_dict.keys():
                pairs_dict[c] = 1
            else:
                pairs_dict[c] += 1
        return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

    class_count = (pairs.flatMap(map_pair)  # <-- MAP PHASE (R1)
                   .groupByKey()  # <-- REDUCE PHASE (R1)
                   .flatMap(gather_pairs)
                   .reduceByKey(lambda x, y: x + y))  # <-- REDUCE PHASE (R2)
    return class_count

def class_count_with_partition(pairs):
    def map_pair_partitions(pair):
        pair_split = pair.split(' ')
        return [(int(pair_split[0]), pair_split[1])]

    def gather_pairs_partitions(pairs):
        pairs_dict = {}
        for p in pairs:
            key, cls = p[0], p[1]
            if cls not in pairs_dict.keys():
                pairs_dict[cls] = 1
            else:
                pairs_dict[cls] += 1

        def maxChainLength(arr, n):
            max = 0
            mcl = [1 for i in range(n)]

            for i in range(1, n):
                for j in range(0, i):
                    if arr[i].a > arr[j].b and mcl[i] < mcl[j] + 1:
                        mcl[i] = mcl[j] + 1
            for i in range(n):
                if max < mcl[i]:
                    max = mcl[i]
            return max
        arr = [pairs_dict]
        print('Length of maximum size chain is',maxChainLength(arr, len(arr)))
        return [(cls, pairs_dict[cls]) for cls in pairs_dict.keys()]

    class_count = (pairs.flatMap(map_pair_partitions) # <-- MAP PHASE (R1)
        .mapPartitions(gather_pairs_partitions)    # <-- REDUCE PHASE (R1)
        .groupByKey()                              # <-- REDUCE PHASE (R2)
		.mapValues(lambda vals: sum(vals)))
    return class_count

def main():

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    assert len(sys.argv) == 3, "Usage: python G30HW1_Minh.py <K> <file_name>"

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # SPARK SETUP
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    conf = SparkConf().setAppName('G30HW1').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # INPUT READING
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    # Read number of partitions
    K = sys.argv[1]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    # Read input file and subdivide it into K random partitions
    data_path = sys.argv[2]
    assert os.path.isfile(data_path), "File or folder not found"
    pair_strings = sc.textFile(data_path,minPartitions=K).cache()
    pair_strings.repartition(numPartitions=K)

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # SETTING GLOBAL VARIABLES
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    num_pairs = pair_strings.count()
    print("Number of pairs =", num_pairs)

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # CLASS COUNT with groupByKey
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    cc = class_count(pair_strings, K)
    print("VERSION WITH DETERMINISTIC PARTITIONS")
    print("Output Pairs =", ' '.join(map(lambda pair: "({},{})".format(pair[0], str(pair[1])), cc.sortByKey().collect())))
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # IMPROVED CLASS COUNT with mapPartitions
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    ccwp = class_count_with_partition(pair_strings)
    print("VERSION WITH SPARK PARTITIONS")
    print("Most frequent class =", ccwp.max(key=lambda x: x[1]))
    print("Max partition size =",)

if __name__ == "__main__":
    main()
