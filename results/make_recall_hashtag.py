"""
Randomly sample 20 tweets from for recall calculation
"""
import random

file = '../outputs/openie/hashtag_emo-openie.op'
output = '../results/recall_hashtag-emo.txt'

with open(file, "r") as f:
    tweets = f.read().split('\n\n')
sample_tweets = random.sample(tweets, 20)

with open(output, "w") as out:
    for tweet in sample_tweets:
        print(tweet.split('\n')[0], file=out)

