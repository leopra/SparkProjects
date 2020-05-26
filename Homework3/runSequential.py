#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# METHOD runSequential
# Sequential 2-approximation for diversity maximization based on matching
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# Import Packages
import math

# Compute the squared euclidean distance (to avoid a sqrt computation)
def squared_euclidean_dist(p, q):
    tmp = 0
    for i in range(0, len(p)-1):
        tmp = tmp + (p[i]-q[i])**2
    return tmp


# runSequential receives a list of tuples and an integer k.
# It comptues a 2-approximation of k points for diversity maximization
# based on matching.
def runSequential(points, k):

    n = len(points)
    if k >= n:
        return points

    result = list()
    candidates = [True for i in range(0,n)]
    
    # find k/2 pairs that maximize distances
    for iter in range(int(k / 2)):
        maxDist = 0.0
        maxI = 0
        maxJ = 0
        for i in range(n):
            if candidates[i] == True: # Check if i is already a solution
                for j in range(n):
                    if candidates[j] == True: # Check if j is already a solution
                        # use squared euclidean distance to avoid an sqrt computation!
                        d = squared_euclidean_dist(points[i], points[j])
                        if d > maxDist:
                            maxDist = d
                            maxI = i
                            maxJ = j
        result.append( points[maxI] )
        result.append( points[maxJ] )
        candidates[maxI] = False
        candidates[maxJ] = False

    # Add one more point if k is odd: the algorithm just start scanning
    # the input points looking for a point not in the result set.
    if k % 2 != 0:
        for i in range(n):
            if candidates[i] == True:
                result.append( points[i] )
                break

    return result

