import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class G48HM4 {
    public static void main(String[] args) throws Exception {

        //------- PARSING CMD LINE ------------
        // Parameters are:
        // <path to file>, k, L and iter

        if (args.length != 4) {
            System.err.println("USAGE: <filepath> k L iter");
            System.exit(1);
        }
        String inputPath = args[0];
        int k = 0, L = 0, iter = 0;
        try {
            k = Integer.parseInt(args[1]);
            L = Integer.parseInt(args[2]);
            iter = Integer.parseInt(args[3]);
        } catch (Exception e) {
            e.printStackTrace();
        }
        if (k <= 2 && L <= 1 && iter <= 0) {
            System.err.println("Something wrong here...!");
            System.exit(1);
        }
        //------------------------------------
        final int k_fin = k;

        //------- DISABLE LOG MESSAGES
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        //------- SETTING THE SPARK CONTEXT
        SparkConf conf = new SparkConf(true).setAppName("kmedian new approach");
        JavaSparkContext sc = new JavaSparkContext(conf);

        //------- PARSING INPUT FILE ------------
        JavaRDD<Vector> pointset = sc.textFile(args[0], L)
                .map(x -> strToVector(x))
                .repartition(L)
                .cache();
        long N = pointset.count();
        System.out.println("Number of points is : " + N);
        System.out.println("Number of clusters is : " + k);
        System.out.println("Number of parts is : " + L);
        System.out.println("Number of iterations is : " + iter);

        //------- SOLVING THE PROBLEM ------------
        double obj = MR_kmedian(pointset, k, L, iter);
        System.out.println("Objective function is : <" + obj + ">");
    }

    public static Double MR_kmedian(JavaRDD<Vector> pointset, int k, int L, int iter) {
        //
        // --- ADD INSTRUCTIONS TO TAKE AND PRINT TIMES OF ROUNDS 1, 2 and 3
        //

        //------------- ROUND 1 ---------------------------
        double npoints = pointset.count();
        long start;
        System.out.println("ROUND 1");
        start= System.currentTimeMillis();
        JavaRDD<Tuple2<Vector, Long>> coreset = pointset.mapPartitions(x ->
        {
            ArrayList<Vector> points = new ArrayList<>();
            ArrayList<Long> weights = new ArrayList<>();
            while (x.hasNext()) {
                points.add(x.next());
                weights.add(1L);
            }
            ArrayList<Vector> centers = kmeansPP(points, weights, k, iter);
            ArrayList<Long> weight_centers = compute_weights(points, centers);
            ArrayList<Tuple2<Vector, Long>> c_w = new ArrayList<>();
            for (int i = 0; i < centers.size(); ++i) {
                Tuple2<Vector, Long> entry = new Tuple2<>(centers.get(i), weight_centers.get(i));
                c_w.add(i, entry);
            }
            return c_w.iterator();
        });
        coreset.cache().count();
        System.out.println("\nExecution time Round 1: "+ (System.currentTimeMillis()-start));


        //------------- ROUND 2 ---------------------------
        System.out.println("ROUND 2");
        start= System.currentTimeMillis();
        ArrayList<Tuple2<Vector, Long>> elems = new ArrayList<>(k * L);
        elems.addAll(coreset.collect());
        //start= System.currentTimeMillis();
        ArrayList<Vector> coresetPoints = new ArrayList<>();
        ArrayList<Long> weights = new ArrayList<>();
        for (int i = 0; i < elems.size(); ++i) {
            coresetPoints.add(i, elems.get(i)._1);
            weights.add(i, elems.get(i)._2);
        }

        ArrayList<Vector> centers = kmeansPP(coresetPoints, weights, k, iter);
        System.out.println("\nExecution time Round 2: "+ (System.currentTimeMillis()-start));
        //------------- ROUND 3: COMPUTE OBJ FUNCTION --------------------
        //
        //------------- ADD YOUR CODE HERE--------------------------------
        //

        System.out.println("ROUND 3");
        start= System.currentTimeMillis();

        double objective = pointset.map(item ->
        {
            //mapping every point to the distance from their closest center
            double minDist = Math.sqrt(Vectors.sqdist(item, centers.get(0)));
            for (int j = 1; j < centers.size(); j++) {
                double dist = Math.sqrt(Vectors.sqdist(item, centers.get(j)));
                if (dist < minDist)
                    minDist = dist;
            }
            return minDist;
        })
                //computing the sum of the distances
                .reduce((x, y) -> x + y);
        System.out.println("\nExecution time Round 3: "+ (System.currentTimeMillis()-start));
        return objective / npoints;
    }

    public static ArrayList<Long> compute_weights(ArrayList<Vector> points, ArrayList<Vector> centers) {
        Long weights[] = new Long[centers.size()];
        Arrays.fill(weights, 0L);
        for (int i = 0; i < points.size(); ++i) {
            double tmp = euclidean(points.get(i), centers.get(0));
            int mycenter = 0;
            for (int j = 1; j < centers.size(); ++j) {
                if (euclidean(points.get(i), centers.get(j)) < tmp) {
                    mycenter = j;
                    tmp = euclidean(points.get(i), centers.get(j));
                }
            }
            weights[mycenter] += 1L;
        }
        ArrayList<Long> fin_weights = new ArrayList<>(Arrays.asList(weights));
        return fin_weights;
    }

    public static Vector strToVector(String str) {
        String[] tokens = str.split(" ");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    // Euclidean distance
    public static double euclidean(Vector a, Vector b) {
        return Math.sqrt(Vectors.sqdist(a, b));
    }

    private static ArrayList<Vector> kmeansPP(ArrayList<Vector> p, ArrayList<Long> wp, int k, int iter) {
        Random r = new Random();
        ArrayList<Vector> centers = new ArrayList<>();
        boolean[] isCenter = new boolean[p.size()];
        for (int i = 0; i < isCenter.length; i++)
            isCenter[i] = false;
        Vector c0 = Vectors.zeros(p.get(0).size());
        int i1 = r.nextInt(p.size()); // we select at random the first center with uniform probability
        BLAS.copy(p.get(i1), c0);
        centers.add(c0);
        isCenter[i1] = true;

        double[] weightedDistances = new double[p.size()];
        double sumWD = 0; //sum of the weighted distances
        //computing the weighted distance from all the points to the first center
        for (int j = 0; j < p.size(); j++)
            if (!isCenter[j]) {
                double weightDist = Math.sqrt(Vectors.sqdist(centers.get(0), p.get(j))) * wp.get(j);
                // the only centeris the nearest, we consider the weighted distance since we'll need the sum of the weighted distances
                sumWD += weightDist;
                weightedDistances[j] = weightDist;
            }

        for (int i = 1; i < k; i++) {
            //now we selecth the i-th center
            Vector ci = Vectors.zeros(p.get(0).size());
            double threshold = r.nextDouble() * sumWD;
            //the threshold is not just a random number in [0,1] because the weighted distances are not divided by their sum
            int index = 0;
            double sum = 0;
            while (sum <= threshold && index < weightedDistances.length) {
                //weightedDistances[j] =0 where p[j] is a center
                sum += (weightedDistances[index]);
                index++;
            }
            // now p[index] will be our next center chosen with probability w_p*(d_p)/(sum_{q non center} w_q*(d_q))
            BLAS.copy(p.get(index), ci);
            centers.add(ci);
            isCenter[index] = true;

            //once computed the new center we update the wheighted distances between the non-center points and their nearest center and compute the sum for the choice of the (i+1)-th center
            sumWD = 0;
            for (int j = 0; j < p.size(); j++)
                if (!isCenter[j]) {
                    double weightDist = Math.sqrt(Vectors.sqdist(ci, p.get(j))) * wp.get(j);
                    if (weightDist < weightedDistances[j]) //updating the distance if the point is closer to the new center
                        weightedDistances[j] = weightDist;
                    sumWD += weightedDistances[j];
                }
        }

        //  for (int i = 0; i < centers.size(); i++)
        //    System.out.println(centers.get(i));
        //Lloyd refinement
        int round = 0;
        int pointSize = centers.get(0).size();
        int[] clustering; //for each p[i] this array will contain an integer with the cluster containing it
        int[] weightedPointsCluster = new int[centers.size()]; // this array will contain at location i the sum of the weights of the points in the i-th cluster
        while (round < iter) {
            clustering = partition(p, centers); //partitions assigns to all the points p[i] an integer j corresponding to the nearest cluster centers[j]
            for (int i = 0; i < centers.size(); i++) {
                centers.set(i, Vectors.zeros(pointSize));
                weightedPointsCluster[i] = 0;
            }
            //computing centroids as the mean of the points in a cluster
            //computing partial sums
            for (int i = 0; i < p.size(); i++) {
                Vector tmp = Vectors.zeros(pointSize);
                BLAS.copy(centers.get(clustering[i]), tmp);
                BLAS.axpy(wp.get(i), p.get(i), tmp);
                centers.set(clustering[i], tmp);
                weightedPointsCluster[clustering[i]] += wp.get(i);

            }
            //computing means
            for (int i = 0; i < centers.size(); i++) {
                Vector tmp = Vectors.zeros(pointSize);
                BLAS.copy(centers.get(i), tmp);
                if (weightedPointsCluster[i] != 0)
                    BLAS.scal(1d / weightedPointsCluster[i], tmp);

                centers.set(i, tmp);
            }
            round++;
        }
        return centers;
    }

    //implements the partition of a set of points around the centers by assigning to each point
    //an integer correspondig to the index of its center in the arraylist of centers
    private static int[] partition(ArrayList<Vector> points, ArrayList<Vector> centers) {
        int[] clustering = new int[points.size()];
        for (int j = 0; j < points.size(); j++) {
            //we compute the nearest center for every point
            double dmin = Double.MAX_VALUE;
            double d;
            int centroid = -1;
            for (int i = 0; i < centers.size(); i++) {
                d = Math.sqrt(Vectors.sqdist(centers.get(i), points.get(j)));
                if (d < dmin) {
                    dmin = d;
                    centroid = i;
                }

            }
            clustering[j] = centroid;

        }
        return clustering;
    }

}
