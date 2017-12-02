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

def find_Centroid(tweets):
    distToAll = lambda x: np.mean(list(map(lambda y : distance.jaccard(x[1],y[1]), tweets)))
    tweetVal = np.argmin(list(map(distToAll, tweets)))
    return tweets[tweetVal]

def find_clusters(kernelList, tweets)
    clusters = [[] for x in range(len(kernelList))]
    for tweet in tweets:
        myclust = np.argmin(list(map(lambda x:distance.jaccard(x[1], tweet[1]), kernelList)))
        clusters[myclust].append(tweet)
