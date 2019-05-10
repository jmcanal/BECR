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

    # emo_matrix = np.load('../lib/emotion_lexicon/NRC/nrc_emotion_lexicon_matrix.npy')
    # w2idx = pickle.load(open('../lib/emotion_lexicon/NRC/nrc_word_map.pkl', "rb"))
    # emo2idx = pickle.load(open('../lib/emotion_lexicon/NRC/nrc_emotion_map.pkl', "rb"))

    emo_list = pickle.load(open('../lib/emotion_lexicon/emotion_kw_list/emotion_keywords_syns.pkl', "rb"))
    # emo_list = pickle.load(open('../lib/emotion_lexicon/emotion_kw_list/emotion_keywords.pkl', "rb"))

    def __init__(self, tweet, tags):
        """
        Initialize by saving the tweet and tags
        :param tweet: the given tweet
        :param tags: pos tags for the tweet
        """
        # self.tweet = tweet
        # self.words = self.tweet.split()
        self.tagged = tags
        self.words = [t[0] for t in tags]
        tweet_info = tweet.split("\t")
        self.tweet = tweet_info[13]
        self.source = tweet_info[14]
        self.emotions = tweet_info[15].split(' or ')
        self.other_emotion = tweet_info[16]
        self.target = tweet_info[19]
        self.emo_word = tweet_info[20]
        self.cause = tweet_info[21]

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

        # exclude POSITIVE or NEGATIVE for nrc
        # exclude_list = [5, 6]

        # exclude for dm
        # if max_val < .5:
        #     return False
        exclude_list = []

        return val == max_val and n not in exclude_list

    def get_emotions(self):
        """
        Get emotional words associated with this tweet
        :return:
        """
        word_emos = []
        for word in self.words:
            tag = [tag[1] for tag in self.tagged if tag[0] == word][0]

            # filter by keyword
            # V and A for hashtags, VB and JJ for semeval
            pos_list = ['V', 'A', 'VB', 'JJ']
            if word in self.emo_list and tag in pos_list:
                word_emos.append(word)


            # depechemood lexicon has tagged words
            # word = word + "#" + tag.lower()
            #
            # if word in self.w2idx.keys():
            #     emo_idx_list = self.emo_matrix[self.w2idx[word]]
            #     emo_list = [self.get_emotion(n) for n, val in enumerate(emo_idx_list) if self.should_include(emo_idx_list, n, val)]
            #
            #     # V and A for hashtags, VB and JJ for semeval
            #     pos_list = ['V', 'A', 'VB', 'JJ']
            #     if emo_list and (tag in pos_list):
            #         emo_list = word + ": " + ", ".join(emo_list)
            #         word_emos.append(emo_list)
        return word_emos

class TweetFile:

    def __init__(self, original_file, raw_tweets):
        """
        Initialize by saving a list of the raw tweets
        :param raw_tweets: tweet file
        """
        tweets = [line for line in open(raw_tweets, "r")]
        self.tags = CMUTweetTagger.runtagger_parse(tweets)
        self.tweets = [line for line in open(original_file, "r")][1:]

    def filter_tweets(self, debug_file):
        """
        Filter tweets that contain emotion words
        :return: string of tweets containing emotion words
        """
        tweet_list = []
        with open(debug_file, 'a+') as d:
            for index, line in enumerate(self.tweets):
                tweet = Tweet(line, self.tags[index])
                if tweet.get_emotions():
                    print("TWEET", tweet.tweet, file=d)
                    print("WORD(S):", file=d)
                    for item in tweet.get_emotions():
                        print(item, file=d)
                    print("EMOTIONS", ", ".join(tweet.emotions), file=d)
                    print("OTHER EMOTION", tweet.other_emotion, file=d)
                    print("CLUE", tweet.emo_word, file=d)
                    print("SOURCE", tweet.source, file=d)
                    print("TARGET", tweet.target, file=d)
                    print("CAUSE", tweet.cause, file=d)
                    print("\n", file=d)
                    tweet_list.append(tweet.tweet.rstrip())
        return "\n".join(tweet_list)


def main():
    """
    Filter tweets by emotion lexicon
    :return: void
    """

    original_tweets = sys.argv[1]
    raw_tweets = sys.argv[2]
    output_file = sys.argv[3]
    debug_file = sys.argv[4]

    tf = TweetFile(original_tweets, raw_tweets)
    with open(output_file, 'a+') as out:
        print(tf.filter_tweets(debug_file), file=out)

if __name__ == "__main__":
    main()
