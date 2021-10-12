import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd

f=open("Info.txt", "w")

class Instance():
    def __init__(self,id,attributes):
        self.id=id
        self.attributes=attributes

    def getAttrVector(self):
        return self.attributes

    def getId(self):
        return self.id

class Cluster():


    def __init__(self, longV):
        self.averageVector=[0]*longV
        self.instaces = []


    def addIntances(self,instance):

        self.instaces.append(instance)
        iArr=instance.getAttrVector()
        for i in range(0,len(self.averageVector)):
            self.averageVector[i]+=iArr[i]

    def getCentroid(self):
        numIns=len(self.instaces)
        centroid=self.averageVector
        cent=[0]*len(centroid)
        for j in range(0,len(self.averageVector)):
            cent[j]=centroid[j]/numIns
        return cent

    def getInstances(self):
        return self.instaces

def mergeClusters(c1,c2):

    c1i=c1.getInstances()
    c2i=c2.getInstances()
    c3 = Cluster(len(c1i[0].getAttrVector()))

    for i in range(0,len(c1i)):
        c3.addIntances(c1i[i])
    for j in range(0,len(c2i)):
        c3.addIntances(c2i[j])

    return c3

#Clase instancia que guarda el numero de instancia y el vector de atributos


def clusterDistance(c1,c2):
    "Average-Link"
    cc1=c1.getCentroid()
    cc2=c2.getCentroid()
    return distance.minkowski(cc1,cc2,1)

def flatVector(vec):
    flat_list = []
    original_list = vec
    for l in original_list:
        for item in l:
            flat_list.append(item)

    return flat_list

def findMin(distanceMatrix):
    min=distanceMatrix[0,1]
    posI=0
    posJ=1
    for i in range(0,len(distanceMatrix)):
        for j in range(i+1,len(distanceMatrix)):
            if min> distanceMatrix[i,j]:
                min=distanceMatrix[i,j]
                posI=i
                posJ=j

    return [posI,posJ]

def createDistanceMatrix(clustersArray):
    #Creamos una matrix vacia de NxN
    dMatrix=np.zeros([len(clustersArray),len(clustersArray)])
    #Solo vamos a calcular el triangulo derecho superior para no duplicar calculos
    for i in range(0,len(clustersArray)):
        for j in range(i+1,len(clustersArray)):
            dMatrix[i,j]=clusterDistance(clustersArray[i],clustersArray[j])

    return dMatrix

def deleteClusters(distanceMatrix,i,j):
    if i>j:
        distanceMatrix=np.delete(distanceMatrix,i,0)
        distanceMatrix = np.delete(distanceMatrix, i, 1)
        distanceMatrix = np.delete(distanceMatrix, j, 0)
        distanceMatrix = np.delete(distanceMatrix, j, 1)
    else:
        distanceMatrix = np.delete(distanceMatrix, j, 0)
        distanceMatrix = np.delete(distanceMatrix, j, 1)
        distanceMatrix = np.delete(distanceMatrix, i, 0)
        distanceMatrix = np.delete(distanceMatrix, i, 1)

    return distanceMatrix

def printableCluster(clustersArray):
    vec=" ;"
    for i in range(0,len(clustersArray)):
        ins=clustersArray[i].getInstances()
        vec+="{"
        for j in range(0,len(ins)):
            if j!=0:vec+=","
            vec+=str(ins[j].getId())
        vec+="};"

    return vec

def clustersToArray(nInstances,clustersArray):
    rst=np.zeros(nInstances, dtype=int)
    for i in range(0,len(clustersArray)):
        instArr=clustersArray[i].getInstances()

        for j in range(0,len(instArr)):
            rst[instArr[j].getId()]=i


    return rst


def HierarchicalClustering(n_clusters,instances):
 clusterArray=[]
 nInstances=len(instances)
 #Añadimos las instancias a un cluster cada una
 objs=list()
 for x in range(0,nInstances):
     #f.write("Añadiendo Instancia: "+str(x))
     inst=Instance(x,instances[x])
     c=Cluster(len(instances[x]))
     c.addIntances(inst)

     clusterArray.append(c)



 #Calculamos las matriz de distancia entre clusters
 distMatrix=createDistanceMatrix(clusterArray)

 itr=0
 #Empieza a iterar hasta que obtengamos el numero de clusters deseados
 while len(clusterArray)>n_clusters:

     itr+=1

     #Usamos la matriz de distancias para calcular los 2 clusters mas proximos
     minPos=findMin(distMatrix)
     dMik=distMatrix[minPos[0],minPos[1]]
     c1=clusterArray[minPos[0]]
     c2 = clusterArray[minPos[1]]
     #Creamos el cluster c1 U c2
     c3=mergeClusters(c1,c2)
     #Ahora tenemos que eliminar c1 y c2 de clusterArray y de la distanceMatrix

     del clusterArray[minPos[1]]
     del clusterArray[minPos[0]]
     distMatrix=deleteClusters(distMatrix,minPos[0],minPos[1])

     #Añadimos el cluster a clusterArray
     clusterArray.append(c3)

     #Por ultimo añadimos el nuevo cluster a distanceMatrix y calculamos las distancias de ese nuevo cluster
     # Insertar fila
     distMatrix = np.insert(distMatrix, distMatrix.shape[0], np.zeros(len(distMatrix)), 0)
     # Insertar columna
     distMatrix = np.insert(distMatrix, distMatrix.shape[1], np.zeros(len(distMatrix)), 1)

     #Calcular las nuevas distancias y las añadimos

     for i in range(0,len(distMatrix)):

         distMatrix[i,len(distMatrix)-1]=clusterDistance(clusterArray[i],c3)
     #Escribimo en el fichero de ClusteringInfo
     f.write("Iteration= "+str(itr)+", NumClusters: "+str(len(clusterArray))+", Silhouette: ?,"+", DisMikwoski used in Iteration: "+str(dMik)+printableCluster(clusterArray))
     f.write("\n")

 f.close()
 rst=clusterArray

 return rst


def printMatrix(A):
   f.write(str(A))



def plotClustering(instances,clusterArray):
    plt.figure(figsize=(10, 7))
    plt.scatter(instances[:, 0], instances[:, 1], c=clusterArray, cmap='rainbow')
    plt.show()

def saveModel(textsArray, attrArray,clusterLabels ):
    data={
        "Text":textsArray,
        "Attributes":attrArray,
        "Cluster":clusterLabels
    }
    df=pd.DataFrame(data,columns=["Text","Attributes","Cluster"])
    df.to_csv("Model.csv")
    return True

def loadModel(modelPath):
    return True

def calculateInstance(model, instance):
    return True
#ESTRUCTURA RECIBIDA--> Array con ["Texto","Vector con texto convertido en numeros"
X = np.array([
                    [5, 3],
                  [10, 15],
                  [15, 12],
                  [24, 10],
                  [30, 30],
                  [85, 70],
                  [71, 80],
                [60, 78],
                  [70, 55],
                  [80, 91], ])
intanceText=np.array(["ONE",
                  "TWO",
                  "THREE",
                  "FOUR",
                  "FIVE",
                  "SIX",
                  "SEVEN",
                  "EIGHT",
                  "NINE",
                  "TEN",])

cluster=HierarchicalClustering(2,X)

labels=clustersToArray(len(X), cluster)
np.savetxt("prueba.txt",intanceText,delimiter=",")
#saveModel(intanceText,X,labels)
#plotClustering(X,cluster)
