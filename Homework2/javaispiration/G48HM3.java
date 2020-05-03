import org.apache.spark.mllib.linalg.*;
import org.apache.spark.mllib.linalg.Vector;
import java.util.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

public class G48HM3 {

    public static void main(String[] args) throws IOException {
        ArrayList<Vector> points = readVectorsSeq("covtype.data");
        ArrayList<Long> weights = new ArrayList<>();

        for (int i = 0; i < points.size(); i++)
            weights.add(1L);

        ArrayList<Vector> c = kmeansPP(points, weights, 100, 3);
        for (int i = 0; i < c.size(); i++)
            System.out.println(c.get(i));

        System.out.println("objective function =  " + kmedianObj(points, c));

    }

    private static ArrayList<Vector> kmeansPP(ArrayList<Vector> p, ArrayList<Long> wp, int k, int iter) {
        Random r = new Random();
        ArrayList<Vector> centers = new ArrayList<>();
        boolean[] isCenter = new boolean[p.size()];
        for (int i = 0; i < isCenter.length; i++)
            isCenter[i] = false;
        Vector c0 = Vectors.zeros(p.get(0).size());
        int i1 =  r.nextInt(p.size()); // we select at random the first center with uniform probability
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

        for (int i = 0; i < centers.size(); i++)
            System.out.println(centers.get(i));


        //Lloyd refinement
        int round = 0;
        int pointSize = centers.get(0).size();
        int[] clustering; //for each p[i] this array will contain an integer with the cluster containing it
        int [] weightedPointsCluster= new int[centers.size()]; // this array will contain at location i the sum of the weights of the points in the i-th cluster
        while (round < iter) {
            clustering = partition(p,centers); //partitions assigns to all the points p[i] an integer j corresponding to the nearest cluster centers[j]
            for(int i =0;i<centers.size();i++) {
                centers.set(i, Vectors.zeros(pointSize));
                weightedPointsCluster[i]=0;
            }
            //computing centroids as the mean of the points in a cluster
            //computing partial sums
            for(int i =0;i<p.size();i++){
                Vector tmp=Vectors.zeros(pointSize);
                BLAS.copy(centers.get(clustering[i]),tmp);
                BLAS.axpy(wp.get(i),p.get(i),tmp );
                centers.set(clustering[i],tmp);
                weightedPointsCluster[clustering[i]]+=wp.get(i) ;
            }
            //computing means
            for (int i =0;i<centers.size();i++){
                Vector tmp=Vectors.zeros(pointSize);
                BLAS.copy(centers.get(i),tmp);
                BLAS.scal(1d/weightedPointsCluster[i],tmp );
                centers.set(i,tmp);
            }
            round++;
        }
        return centers;
    }

    private static double kmedianObj(ArrayList<Vector> points, ArrayList<Vector> centers) {
        int[] clustering = partition(points, centers);
        double sum = 0;
        for (int i = 0; i < points.size(); i++)
            sum += Math.sqrt(Vectors.sqdist(points.get(i), centers.get(clustering[i])));
        return sum / points.size();                                                      //todo non confermato
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

    private static Vector strToVector(String str) {
        String[] tokens = str.split(" ");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    private static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }


}
