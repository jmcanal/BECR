"""
Randomly sample 20 tweets from for recall calculation
"""
import random

file = '../data/preprocessed/split/filtered_tweets_test.txt'
output = '../results/baseline/recall_filtered_test_25.txt'

with open(file, "r") as f:
    tweets = f.read().split('\n')
tweets = [t for t in tweets if not ""]
sample_tweets = random.sample(tweets, 25)

with open(output, "w") as out:
    for tweet in sample_tweets:
        print(tweet.split('\n')[0], file=out)

