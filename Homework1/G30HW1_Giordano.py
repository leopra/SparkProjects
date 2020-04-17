from pyspark import SparkContext, SparkConf
import sys
import os


def class_count(pairs, K):
    def map_pair_phase(pair):
        pair_split = pair.split(' ', 1)
        return int(pair_split[0]) % K, pair_split[1]  # Class Count maps input pairs one-to-one

    def gather_pairs(pairs):
        pairs_dict = {}
        for c in pairs[1]:
            if c not in pairs_dict.keys():
                pairs_dict[c] = 1
            else:
                pairs_dict[c] += 1
        return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

    class_count = (pairs.map(map_pair_phase)  # <-- MAP PHASE (R1), map function is designed for one-to-one maps as opposed to flatMap (one-to-many)
                   .groupByKey()  # <-- REDUCE PHASE (R1)
                   .flatMap(gather_pairs)
                   .reduceByKey(lambda x, y: x + y))  # <-- REDUCE PHASE (R2)
    return class_count


def class_count_with_partition(pairs):
    def map_pair_phase(pair):
        pair_split = pair.split(' ', 1)
        return int(pair_split[0]), pair_split[1]  # Class Count maps input pairs one-to-one

    def gather_pairs_partitions(pairs):
        pairs_dict = {}
        for p in pairs:
            c = p[1]
            if c not in pairs_dict.keys():
                pairs_dict[c] = 1
            else:
                pairs_dict[c] += 1
        pairs_dict["maxPartitionSize"] = sum(pairs_dict.values())  # Special pair maxPartitionSize is produced for each partition
        return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

    class_pairs = (pairs.map(map_pair_phase)  # <-- MAP PHASE (R1)
                   .mapPartitions(gather_pairs_partitions))  # <-- REDUCE PHASE (R1)

    # R2 implementation is split in two to manage different reductions: sum of partial class counts and max of partition sizes
    class_count = class_pairs.filter(lambda x: x[0] != "maxPartitionSize").reduceByKey(lambda x, y: x + y)  # <-- REDUCE PHASE (R2)
    max_partition_size = class_pairs.filter(lambda x: x[0] == "maxPartitionSize").reduceByKey(lambda x, y: max(x, y))  # <-- REDUCE PHASE (R2)

    return class_count.union(max_partition_size)  # This generates a single RDD that includes all the required pairs


def main():
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    assert len(sys.argv) == 3, "Usage: python G30HW1.py <K> <file_name>"

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
    pair_strings = sc.textFile(data_path, minPartitions=K).cache()
    pair_strings.repartition(numPartitions=K)

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # SETTING GLOBAL VARIABLES
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    num_pairs = pair_strings.count()
    print("Number of pairs = ", num_pairs)

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # CLASS COUNT with groupByKey
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    cc = class_count(pair_strings, K)
    pair_list = cc.sortByKey().collect()
    print("VERSION WITH DETERMINISTIC PARTITIONS")
    print(f"Output Pairs = {' '.join([str(pair) for pair in pair_list])}")  # List comprehension allows for easy joining of strings and printing

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # CLASS COUNT with mapPartitions
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    cc = class_count_with_partition(pair_strings)

    # The map function, in case of ties, picks the first maximum value
    # Sorting by key before searching the maximum guarantees that the map function will pick the smaller class in
    # alphabetical order
    max_count = cc.filter(lambda x: x[0] != "maxPartitionSize").sortByKey().max(lambda pair: pair[1])  # Again, statistics computation is split due to the different purposes of pairs
    max_partition_size = cc.filter(lambda x: x[0] == "maxPartitionSize").take(1)
    print("VERSION WITH SPARK PARTITIONS")
    print(f"Most frequent class = {max_count}")
    print(f"Max partition size = {max_partition_size[0][1]}")  # take function returns a List of pairs, so [0][1] is used to access the final value since there is only one maxPartitionSize pair


if __name__ == "__main__":
    main()
