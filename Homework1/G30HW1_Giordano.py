from pyspark import SparkContext, SparkConf
import sys
import os


def class_count(pairs, K):
    def map_pair_phase(pair):
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

    class_count = (pairs.flatMap(map_pair_phase)  # <-- MAP PHASE (R1)
                   .groupByKey()  # <-- REDUCE PHASE (R1)
                   .flatMap(gather_pairs)
                   .reduceByKey(lambda x, y: x + y))  # <-- REDUCE PHASE (R2)
    return class_count


def class_count_with_partition(pairs):
    def map_pair_phase(pair):
        pair_split = pair.split(' ', 1)
        return [(int(pair_split[0]), pair_split[1])]

    def gather_pairs_partitions(pairs):
        pairs_dict = {}
        i = 0
        for p in pairs:
            i += 1
            c = p[1]
            if c not in pairs_dict.keys():
                pairs_dict[c] = 1
            else:
                pairs_dict[c] += 1
        pairs_dict["maxPartitionSize"] = i
        return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

    class_init = (pairs.flatMap(map_pair_phase)  # <-- MAP PHASE (R1)
                  .mapPartitions(gather_pairs_partitions)  # <-- REDUCE PHASE (R1)
                  .groupByKey())  # <-- REDUCE PHASE (R2)

    class_count = class_init.filter(lambda x: x[0] != "maxPartitionSize").mapValues(sum)
    max_partition_size = class_init.filter(lambda x: x[0] == "maxPartitionSize").mapValues(max)

    return class_count.union(max_partition_size)


def main():
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    assert len(sys.argv) == 3, "Usage: python G30HW1_Giordano.py <K> <file_name>"

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
    print("VERSION WITH DETERMINISTIC PARTITIONS")
    print("Output Pairs = ", ' '.join(map(lambda pair: f"({pair[0]},{pair[1]})", cc.sortByKey().collect())))

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # CLASS COUNT with mapPartitions
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    cc = class_count_with_partition(pair_strings)
    # The map function, in case of ties, picks the first maximum value
    # Sorting by key before searching the maximum guarantees that the map function will pick the smaller class in
    # alphabetical order
    max_count = cc.sortByKey().filter(lambda x: x[0] != "maxPartitionSize").max(lambda pair: pair[1])
    max_partition_size = cc.filter(lambda x: x[0] == "maxPartitionSize").collect()
    print("VERSION WITH SPARK PARTITIONS")
    print(f"Most frequent class = ({max_count[0]},{max_count[1]})")
    print(f"Max partition size = {max_partition_size[0][1]}")


if __name__ == "__main__":
    main()
