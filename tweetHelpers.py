import distance
import json
import numpy as np
import nltk

def load_tweets(path):
   fileptr = open(path, "r", 1)
   deeta = []
   with fileptr as f:
       for line in f:
           deeta.append(json.loads(line))
   fileptr.close()
   return [(y,nltk.word_tokenize(x["text"])) for x,y in zip(deeta, range(len(deeta)))]

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

#on occassion, you'll centroids which, though different tweets, have the exact same content. when
#that happens, one of your kernels will be assigned an empty cluster.
#I deal with that by just leaving that cluster empty. hopefully, all the other clusters will
#eventually move out of the way and leave that cluster to itself.
#some code to test it:
tweeties = load_tweets("Tweets.json")
klusters = [tweeties[np.random.randint(len(tweeties))] for x in range(250)]
for i in range(20):
    grups = find_clusters(klusters, tweeties)
    klusters = [find_new_centroid(c,x) for c,x in zip(klusters,grups)]
    print("current sse:", k_means_sse(unzip(klusters,1), [unzip(x,1) for x in grups], distance.jaccard))
