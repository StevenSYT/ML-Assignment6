import distance
import json
import numpy as np
import nltk
import functools
import csv
import sys
def load_tweets(path):
    fileptr = open(path, "r", 1)
    deeta = []
    with fileptr as f:
        for line in f:
            deeta.append(json.loads(line))
    fileptr.close()
    return [(y,nltk.word_tokenize(x["text"]), x["id"]) for x,y in zip(deeta, range(len(deeta)))]

#returns the centroid of a list of tweets
#arguments:
#tweets: a list of tokenized tweets.
def find_new_centroid(centroid, tweets):
    if not tweets:
        return centroid
    distToAll = lambda x: np.mean(list(map(lambda y : distance.jaccard(x[1],y[1]), tweets)))
    tweetVal = np.argmin(list(map(distToAll, tweets)))
    return tweets[tweetVal]

#finds the clusters associated with the centroids in centroidList
#arguments:
#centroidList: a list of n tokenised tweets.
#tweets: a list of tokenised tweets
#returns:
#n lists of tokenised tweets.
def find_clusters(centroidList, tweets):
    clusters = [[] for x in range(len(centroidList))]
    for tweet in tweets:
        myclust = np.argmin(list(map(lambda x:distance.jaccard(x[1], tweet[1]), centroidList)))
        clusters[myclust].append(tweet)
    return clusters

#finds the sse of a clustering result.
#arguments:
#centroids: a list of centroids.
#clusters: a list of lists, of instances.
#distance metric: the distance metric to compare the centroids to the points.
def k_means_sse(centroids, clusters, distance_metric):
    def sum_distances(c, clust):
        if not clust:
            return 0
        return sum(map(lambda x: pow(distance_metric(c, x),2), clust))
    return sum(map(sum_distances, centroids, clusters))

def unzip(twople_list, val):
    return [x[val] for x in twople_list]

def print_cluster_data(centroids,clusters,outputFile,distance_metric, numIterations):
    outFile = open(outputFile, "w+")
    def bracketless_list(z):
        return str(functools.reduce(lambda x,y: str(x)+","+str(y), z))
    for i,c,l in zip(range(len(centroids)),centroids,clusters):
        if not l:
            outFile.write(str(i)+" ("+str(c[2])+")"+"\t(empty)")
        else:
            l = unzip(l,2)
            outFile.write(str(i)+" ("+str(c[2])+")"+"\t"+bracketless_list(l))
        outFile.write("\n\n")
    outFile.write("SSE: "+str(k_means_sse(unzip(centroids,1), [unzip(x,1) for x in clusters],
         distance_metric)))
    outFile.write("\nConverged in "+str(numIterations)+" iterations")
    outFile.close()

def initialise_centroids_from_IDs(tweets, IDs):
    c = []
    for idd in IDs:
        for tweet in tweets:
            if tweet[2] == idd:
                c.append(tweet)
    return c
def k_means_iteration(centroids, data):
    grups = find_clusters(centroids, data)
    centroids = [find_new_centroid(c,x) for c,x in zip(centroids,grups)]
    return centroids, grups

#on occassion, you'll centroids which, though different tweets, have the exact same content. when
#that happens, one of your kernels will be assigned an empty cluster.
#I deal with that by just leaving that cluster empty. hopefully, all the other clusters will
#eventually move out of the way and leave that cluster to itself.
def runme(tweetsFile, outputFile, seedsFile = None, numClusts = 25):
    tweets = load_tweets(tweetsFile)
    #load seeds from file
    if seedsFile == None:
        centroids= [tweets[np.random.randint(len(tweets))] for x in range(numClusts)]
    else:
        IDs = []
        fileptr = open(seedsFile, "r", 1) 
        readlist = list(csv.reader(fileptr))
        for x in readlist:
            IDs.append(int(x[0]))
        centroids = initialise_centroids_from_IDs(tweets, IDs)
        numClusts = len(centroids)
        fileptr.close()
    #run K-means until convergence
    centroidsold = centroids
    centroids, clusters = k_means_iteration(centroidsold, tweets)
    i = 0 
    while (centroidsold != centroids):
        i +=1
        centroidsold = centroids
        centroids, clusters = k_means_iteration(centroidsold, tweets)
    #k- means has now converged.
    print_cluster_data(centroids, clusters, outputFile, distance.jaccard, i)

#actual running code
if len(sys.argv) < 4:
    print("please use all args")
    exit()
numClusters = sys.argv[1] 
seeds = sys.argv[2]
tweets = sys.argv[3]
out = sys.argv[4]
runme(tweets,out,seeds,numClusters)
