import pandas as pd
import numpy as np
import random
import math

data = pd.read_csv("http://www.utdallas.edu/~axn112530/cs6375/unsupervised/test_data.txt", sep=("\t"))

data.drop(["id"], axis = 1, inplace=True)
clusterNum =  5
print("length of data: ", len(data))
# generate the random indices:
centerIndices = random.sample(range(len(data)+1), clusterNum)
clusters = []


# iloc[[index],[col_num]] keeps dataframe type
print(pd.concat([data.iloc[[0], :], data.iloc[[1], :]]))
for i, item in enumerate(centerIndices):
   clusters.append(data.iloc[[item],:])
   print(clusters[i])

def clusterize(row):
   minDistance = float("inf")
   for index, cluster in enumerate(clusters):
      distance = math.sqrt((cluster.iloc[0]['x']-row['x']) ** 2 + (cluster.iloc[0]['y']-row['y']) ** 2)
      if (distance < minDistance):
         minDistance = distance
         targetIndex = index
      print("the distance to center ",index," is ", distance)
   print("this point goes to the cluster", targetIndex)
   clusters[targetIndex] = pd.concat([clusters[targetIndex], row])


# print(data.head())
# print()
# print("The centerIndices are ",centerIndices)
def kMeans(data):
   for index, row in data.iterrows():
      print("working on point ", index, "...")
      clusterize(data.iloc[[index], :])
