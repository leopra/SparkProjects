import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import scala.Tuple2;

import java.util.*;

public class G48HM2 {

    public static void main(String[] args){
        //spark setup
        System.setProperty("hadoop.home.dir", "d:\\SVILUPPO\\BDC1819\\");
        SparkConf configuration =
                new SparkConf(true)
                        .setAppName("wordCount")
                        .setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(configuration);
        System.out.println("Enter number of partitions for Improved Word Count 2");
        Scanner scanIn = new Scanner(System.in);
        int numPart = scanIn.nextInt();
        //reading from file, if the number of partitions is not specified Spark automatically splits the RDD into 2 partitions
        //we chose to split the RDD into k parts at the beginning in order to avoid the shuffle due to the repartition()
        // function in the last algorithm.
        JavaRDD<String> docs = sc.textFile("documents.txt", numPart);
        docs.count();
        double mean;
        long ts1, ts2, ts3;
        /* The average length of the distinct words is computed for all the algorithms to show that they actually output the same result,
         * the running time is computed excluding the operations for computing the average so to not influence the analysis of the running time
         * of the algorithms with the sum of all the values in the last RDD*/
        long start = System.currentTimeMillis();
        //improved wordcount 1
        JavaDoubleRDD distinctRdd = docs
                // Map phase
                .flatMapToPair((document) -> {
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                // Reduce phase
                .groupByKey()
                .mapValues((it) -> {
                    long sum = 0;
                    for (long c : it) {
                        sum += c;
                    }
                    return sum;
                })
                .mapToDouble((t) -> t._1().length());
        long end = System.currentTimeMillis();

        mean = distinctRdd.sum() / distinctRdd.count();
        System.out.print("MEAN improved word count 1 " + mean + "\n");
        ts1 = end - start;
        start = System.currentTimeMillis();
        //improved word count 2 v1
        distinctRdd = docs
                //round 1 - map, the first map is the same as the first version
                .flatMapToPair((document) -> {
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .groupBy((x) -> {
                    //partitioning
                    Random r = new Random();
                    return r.nextInt(numPart);
                }, numPart)
                //round1 -reduce
                .flatMapToPair((kvp) -> {
                    //kvp represents a key-value pair where the key is assigned by the partitioning procedure
                    // and the value is a collection of Tuple2<String, Long> to which the same key was assigned.
                    // For every word the intermediate count of occurrences in every "document split" of every partition is obtained in the same way as in the map phase
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    HashMap<String, Long> counts = new HashMap<>();
                    for (Tuple2<String, Long> t : kvp._2()) {
                        counts.put(t._1(), t._2() + counts.getOrDefault(t._1(), 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<String, Long>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                //round 2 map: identity
                //round2 reduce
                .groupByKey()
                .mapValues((it) -> {
                    long sum = 0;
                    for (long c : it) {
                        sum += c;
                    }
                    return sum;
                })
                .mapToDouble((t) -> t._1().length());
        end = System.currentTimeMillis();
        mean = distinctRdd.sum() / distinctRdd.count();
        System.out.print("MEAN improved word count 2 v1 " + mean + "\n");
        ts2 = end - start;
        start = System.currentTimeMillis();
        //improved word count v2
        distinctRdd = docs
                //round 1 - map, the first map is the same as the first two versions
                .flatMapToPair((document) -> {
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                //round 1 -reduce
                .mapPartitionsToPair((iterator) -> {//Every partition is associated to an iterator
                    //The intermediate word count is computed as before
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    HashMap<String, Long> counts = new HashMap<>();
                    while (iterator.hasNext()) {
                        Tuple2<String, Long> t = iterator.next();
                        counts.put(t._1(), t._2() + counts.getOrDefault(t._1(), 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<String, Long>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                //round 2 - map: identity
                //round 2 - reduce
                .reduceByKey((x, y) -> x + y) //The transitive and associative function is applied to the values
                .mapToDouble((t) -> t._1().length());
        end = System.currentTimeMillis();
        mean = distinctRdd.sum() / distinctRdd.count();
        ts3 = end - start;
        System.out.print("MEAN improved word count 2 v2 " + mean + "\n");
        System.out.println("Elapsed time for improved word count 1 " + ts1 + " ms");
        System.out.println("Elapsed time for improved word count 2 v1 " + ts2 + " ms");
        System.out.println("Elapsed time for improved word count 2 v2 " + ts3 + " ms");
    }
}
