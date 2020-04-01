from pyspark import SparkContext, SparkConf
import sys
import os


def class_count_1(pairs, K):

    def map_pair_phase_1(pair):
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

    class_count = (pairs.flatMap(map_pair_phase_1)  # <-- MAP PHASE (R1)
                   .groupByKey()  # <-- REDUCE PHASE (R1)
                   .flatMap(gather_pairs)
                   .reduceByKey(lambda x, y: x + y))  # <-- REDUCE PHASE (R2)
    return class_count


def class_count_1_with_partition(docs):

    def map_pair_phase_1(pair):
        pair_split = pair.split(' ', 1)
        return [(int(pair_split[0]), pair_split[1])]

    def gather_pairs_partitions(pairs):
        pairs_dict = {}
        for p in pairs:
            word, occurrences = p[0], p[1]
            if word not in pairs_dict.keys():
                pairs_dict[word] = occurrences
            else:
                pairs_dict[word] += occurrences
        return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

    word_count = (docs.flatMap(map_pair_phase_1)  # <-- MAP PHASE (R1)
                  .mapPartitions(gather_pairs_partitions)  # <-- REDUCE PHASE (R1)
                  .groupByKey()  # <-- REDUCE PHASE (R2)
                  .mapValues(lambda vals: sum(vals)))

    return word_count


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

    cc = class_count_1(pair_strings, K)
    print("VERSION WITH DETERMINISTIC PARTITIONS")
    print("Output Pairs = ", ' '.join(map(lambda pair: "({},{})".format(pair[0], str(pair[1])), cc.sortByKey().collect())))

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # IMPROVED WORD COUNT with mapPartitions
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # wc = word_count_2_with_partition(pair_strings)
    # print("Number of distinct words in the documents = ", wc.count())

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # COMPUTE AVERAGE WORD LENGTH
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # average_word_len = wc.keys().map(lambda x: (x, len(x))).values().mean()
    # print("Average word length = ", average_word_len)


if __name__ == "__main__":
    main()
