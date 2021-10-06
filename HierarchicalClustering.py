import numpy as np
from scipy.spatial import distance







def clusterDistance(c1,c2):
    return distance.minkowski(c1,c2, 1)

def HierarchicalClustering(n_clusters,instances):
 clusters=[]
 for x in len(instances):
     clusters+=[x,instances[x]]

 newClusters=[]
 while len(clusters)>n_clusters:
     evaluated=set({})
     for i in len(clusters)-1:
        if evaluated.__contains__(i)==False:
            minDis=10000000000
            inst=i

            for j in range(1,len(clusters)-1):
                if evaluated.__contains__(j)==False:
                    disM=clusterDistance(clusters[i][1],clusters[j][1])
                    if disM<minDis:
                        minDis=disM
                        inst=j
        evaluated+=i
        evaluated+=j


def plotClustering(instances,clusterArray):

X = np.array([[5, 3],
                  [10, 15],
                  [15, 12],
                  [24, 10],
                  [30, 30],
                  [85, 70],
                  [71, 80],
                  [60, 78],
                  [70, 55],
                  [80, 91], ])
print(clusterDistance(X[1], X[2]))