"""
Filters tweets by verbs and adjectives in the emotion lexicon
"""

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pickle
from lib.tweet_tagger import CMUTweetTagger
import time


class Tweet:

    # emo_matrix = np.load('../lib/emotion_lexicon/DepecheMood/dm_emotion_lexicon_matrix.npy')
    # w2idx = pickle.load(open('../lib/emotion_lexicon/DepecheMood/dm_word_map.pkl', "rb"))
    # emo2idx = pickle.load(open('../lib/emotion_lexicon/DepecheMood/dm_emotion_map.pkl', "rb"))

    emo_matrix = np.load('../lib/emotion_lexicon/NRC/nrc_emotion_lexicon_matrix.npy')
    w2idx = pickle.load(open('../lib/emotion_lexicon/NRC/nrc_word_map.pkl', "rb"))
    emo2idx = pickle.load(open('../lib/emotion_lexicon/NRC/nrc_emotion_map.pkl', "rb"))

    def __init__(self, tweet, tags):
        """
        Initialize by saving the tweet and tags
        :param tweet: the given tweet
        :param tags: pos tags for the tweet
        """
        self.tweet = tweet
        self.tagged = tags
        self.words = [t[0] for t in tags]

    def get_emotion(self, idx):
        """
        Get the emotion from the index
        :param idx: index of the emotion in the mapping
        :return: the emotion string
        """
        return [e for e, i in self.emo2idx.items() if i == idx][0]

    def should_include(self, emo_idx_list, n, val):
        """
        Should this word be included?
        :param emo_idx_list: list of values for each emotion
        :param n: the emotion index
        :param val: the value for the given index
        :return: bool
        """
        max_val = max(emo_idx_list)
        # don't include POSITIVE or NEGATIVE
        return val == max_val and n != 5 and n != 6

    def get_emotions(self):
        """
        Get emotional words associated with this tweet
        :return:
        """
        word_emos = []
        for word in self.words:
            tag = [tag[1] for tag in self.tagged if tag[0] == word][0]

            # depechemood lexicon has tagged words
            # word = word + "#" + tag.lower()

            if word in self.w2idx.keys():
                emo_idx_list = self.emo_matrix[self.w2idx[word]]
                emo_list = [self.get_emotion(n) for n, val in enumerate(emo_idx_list) if self.should_include(emo_idx_list, n, val)]

                # V and A for hashtags, VB and JJ for semeval
                if emo_list and (tag == 'V' or tag == 'A'):
                    emo_list = word + ": " + ", ".join(emo_list)
                    word_emos.append(emo_list)
        return word_emos

class TweetFile:

    def __init__(self, raw_tweets):
        """
        Initialize by saving a list of the raw tweets
        :param raw_tweets: tweet file
        """
        self.tweets = [line for line in open(raw_tweets, "r")]

    def filter_tweets(self, ):
        """
        Filter tweets that contain emotion words
        :return: string of tweets containing emotion words
        """
        tweet_list = []
        start = time.time()
        tags = CMUTweetTagger.runtagger_parse(self.tweets)
        print(time.time() - start)
        with open('emo_debug.txt', 'w') as d:
            for index, line in enumerate(self.tweets):
                tweet = Tweet(line, tags[index])
                if tweet.get_emotions():
                    print("TWEET", tweet.tweet, file=d)
                    print("WORD(S):", file=d)
                    for item in tweet.get_emotions():
                        print(item, file=d)
                    print("\n", file=d)
                    tweet_list.append(tweet.tweet.rstrip())
        return "\n".join(tweet_list)


def main():
    """
    Filter tweets by emotion lexicon
    :return: void
    """

    raw_tweets = sys.argv[1]
    output_file = sys.argv[2]

    tf = TweetFile(raw_tweets)
    with open(output_file, 'w') as out:
        print(tf.filter_tweets(), file=out)

if __name__ == "__main__":
    main()
