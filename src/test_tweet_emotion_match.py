
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pickle
from lib.tweet_tagger import CMUTweetTagger
import time

twitter_file = sys.argv[1]
output_file = sys.argv[2]

emo_matrix = np.load('../lib/emotion_lexicon/DepecheMood/dm_emotion_lexicon_matrix.npy')

w2idx = pickle.load(open('../lib/emotion_lexicon/DepecheMood/dm_word_map.pkl', "rb"))
emo2idx = pickle.load(open('../lib/emotion_lexicon/DepecheMood/dm_emotion_map.pkl', "rb"))


class Tweet:

    def __init__(self, raw, tags):
        self.raw = raw
        self.tagged = tags
        self.words = [t[0] for t in tags]

        self.id = int
        self.topic = self.sentiment = self.emotion = self.tweet_text = ""

        # immediately process tweet
        self.process_tweet()

    def process_tweet(self):
        # formatting for semeval 2016 pt eng dataset
        self.id, self.topic, self.sentiment, self.tweet_text = self.raw.rstrip().split("\t")

        # formatting for hashtag emotion corpus
        # tweet, self.emotion = self.raw.rstrip().split("\t::")
        # tweet_info = tweet.rstrip().split(':\t')
        # self.id = tweet_info[0]
        # self.tweet_text = ' '.join(tweet_info[1:])

    def get_emotion(self, idx):
        return [e for e, i in emo2idx.items() if i == idx][0]

    def should_include(self, emo_idx_list, n, val):
        max_val = max(emo_idx_list)
        return val == max_val and n != 5 and n != 6

    def get_emotions(self):
        word_emos = []
        for word in self.words:
            tag = [tag[1] for tag in self.tagged if tag[0] == word][0]
            tagged_word = word + "#" + tag.lower()
            if tagged_word in w2idx.keys():
                emo_idx_list = emo_matrix[w2idx[tagged_word]]
                emo_list = [self.get_emotion(n) for n, val in enumerate(emo_idx_list) if self.should_include(emo_idx_list, n, val)]

                # V and A for hashtags, VB and JJ for semeval
                if emo_list and (tag == 'V' or tag == 'A'):
                    emo_list = word + ": " + ", ".join(emo_list)
                    word_emos.append(emo_list)
        return word_emos

class TweetFile:

    def __init__(self, output):
        self.output = output

    def get_all_tweets(self):
        tweet_list = []
        lines = [line for line in open(twitter_file, "r")]
        # Format for 2018 files
        # tweets = [' '.join(line.split('\t')[1:-1]) for line in lines]
        # Format for 2016 files
        tweets = [' '.join(line.split('\t')[3:]) for line in lines]
        # print(tweets)
        start = time.time()
        tags = CMUTweetTagger.runtagger_parse(tweets)
        # print(tags)
        print(time.time() - start)
        for index, line in enumerate(lines):
            tweet = Tweet(line, tags[index])
            word_emos = tweet.get_emotions()
            tweet_list.append((tweet.tweet_text, word_emos, tweet.topic))
            # print(tweet_list)
        with open(self.output, "w") as f:
            with open('../data/preprocessed/emotweets_2016_train_testing.txt', 'w') as d:
                tweet_count = 0
                for tweet in tweet_list:
                    if tweet[1]:
                        print(tweet[0], file=d)
                        print("TWEET", tweet[0], file=f)
                        print("TOPIC", tweet[2], file=f)
                        print("WORD(S):", file=f)
                        for item in tweet[1]:
                            print(item, file=f)
                        print("\n", file=f)
                        tweet_count += 1
                print("Total tweets: " + str(tweet_count), file=f)


TweetFile(output_file).get_all_tweets()
