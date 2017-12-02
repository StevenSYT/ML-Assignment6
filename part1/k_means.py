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
   # print("this point goes to the cluster", targetIndex, "with distance ", minDistance)
   clusters[targetIndex] = pd.concat([clusters[targetIndex], row])

def compareCenters(centroids, center):
   for i, centroid in enumerate(centroids):
      if(not centroid.equals(center[i])):
         print("centroid: ")
         print(centroid)
         print("center: ")
         print(center[i])
         print("not fully converged")
         return False
   return True

def kMeans(data, centroids, clusters):
   print("Begin Clustering...")
   iteration = 1
   center = []

   while(iteration <= maxIteration):
      print("iteration", iteration,"...")
      if(center!= []):
         clusters = center
         centroids = center
         center = []
      for index, row in data.iterrows():
         # print("working on point ", index, "...")
         clusterize(data.iloc[[index], :], clusters)

      for index, cluster in enumerate(clusters):
         if (cluster.mean(axis = 0).to_frame().T.equals(cluster.iloc[[0], :])):
            print("centroid", index, "converges")
         center.append(cluster.mean(axis = 0).to_frame().T)
      if(compareCenters(centroids, center)):
         break
      iteration += 1
   print("Clustering finishes!")
   return clusters

def SSE(clusters):
   print("computing SSE... ")
   sseList = []
   for index, cluster in enumerate(clusters):
      sse = 0
      mean = cluster.mean(axis = 0)
      meanX = mean['x']
      meanY = mean['y']
      # print("cluster ", index)
      # print(cluster)
      for i, row in cluster.iterrows():
         sse += ((meanX - row['x']) ** 2 + (meanY - row['y']) ** 2)
      sseList.append(sse)
   return sum(sseList)

def outPut(outPath, clusters, sseResult):
   print("exporting the clusters to file '"+outPath+"'" )
   outFile = open(outPath,"a+")
   for index, cluster in enumerate(clusters):
      outFile.write(str(index)+"\t")
      for i in range(1, len(cluster)):
         outFile.write(str(cluster.index[i]))
         if (i < len(cluster) - 1):
            outFile.write(", ")
      outFile.write("\n")
   outFile.write("SSE = ")
   outFile.write(str(sseResult))
   print("Done!!")
# iloc[[index],[col_num]] keeps dataframe type
# print(pd.concat([data.iloc[[0], :], data.iloc[[1], :]]))

clusterNum =  int(sys.argv[1]) if len(sys.argv)>1 else 5
# print(clusterNum)
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
# print(data.mean(axis = 0).to_frame().T)

for i, item in enumerate(centerIndices):
   clusters.append(data.iloc[[item],:])
   centroids.append(data.iloc[[item],:])
   # print(clusters[i])
print("You decided to use",len(clusters),"clusters")
clusters = kMeans(data, centroids, clusters)
# print("length of clusters is ",len(clusters))
sseResult = SSE(clusters)
outPut(outPath, clusters, sseResult)
