import pandas as pd
import numpy as np
import random
import math
import sys


def clusterize(row, clusters):
   minDistance = float("inf")
   for index, cluster in enumerate(clusters):
      distance = math.sqrt((cluster.iloc[0]['x']-row['x']) ** 2 + (cluster.iloc[0]['y']-row['y']) ** 2)
      if (distance < minDistance):
         minDistance = distance
         targetIndex = index
   print("this point goes to the cluster", targetIndex, "with distance ", minDistance)
   clusters[targetIndex] = pd.concat([clusters[targetIndex], row])

def compareCenters(centroids, center):
   for i, centroids in enumerate(centroids):
      if(not centroids.equals(center[i])):
         return False
   return True

def kMeans(data, centroids, clusters):
   iteration = 1
   center = []

   while(iteration <= maxIteration):
      if(center!= []):
         clusters = center
         centroids = center
         center = []
      for index, row in data.iterrows():
         print("working on point ", index, "...")
         clusterize(data.iloc[[index], :], clusters)

      for index, cluster in enumerate(clusters):
         if (cluster.mean(axis = 0).to_frame().T.equals(cluster.iloc[[0], :])):
            pass
         else:

            center.append(cluster.mean(axis = 0).to_frame().T)
      if(compareCenters(centroids, center)):
         break
      iteration += 1
   return clusters

def SSE(clusters):
   sseList = []
   for index, cluster in enumerate(clusters):
      sse = 0
      mean = cluster.mean(axis = 0)
      meanX = mean['x']
      meanY = mean['y']
      print("cluster ", index)
      print(cluster)
      for i, row in cluster.iterrows():
         sse += ((meanX - row['x']) ** 2 + (meanY - row['y']) ** 2)
      sseList.append(sse)
   return sum(sseList)

def outPut(outPath, clusters, sseResult):
   outFile = open(outPath,"w+")
   for index, cluster in enumerate(clusters):
      outFile.write(str(index)+"\t")
      for i in range(1, len(cluster)):
         outFile.write(str(cluster.index[i]))
         if (i < len(cluster) - 1):
            outFile.write(", ")
      outFile.write("\n")
   outFile.write("SSE = ")
   outFile.write(str(sseResult))
# iloc[[index],[col_num]] keeps dataframe type
# print(pd.concat([data.iloc[[0], :], data.iloc[[1], :]]))

clusterNum =  sys.argv[1] if len(sys.argv)>1 else 5
inPath = sys.argv[2] if len(sys.argv) > 2 else "../data1.txt"
outPath = sys.argv[3] if len(sys.argv) > 3 else "result.txt"
data = pd.read_csv(inPath, sep=("\t"))

data.drop(["id"], axis = 1, inplace=True)
maxIteration = 25
# print("length of data: ", len(data))
# generate the random indices:
centerIndices = random.sample(range(len(data)), clusterNum)
centroids = []
clusters = []
print(data.mean(axis = 0).to_frame().T)

for i, item in enumerate(centerIndices):
   clusters.append(data.iloc[[item],:])
   centroids.append(data.iloc[[item],:])
   # print(clusters[i])
clusters = kMeans(data, centroids, clusters)
# print("length of clusters is ",len(clusters))
sseResult = SSE(clusters)
outPut(outPath, clusters, sseResult)
