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
from src.preprocessing.tweet import Tweet


class TweetFile:

    def __init__(self, tweet_file, filter_by, input_type):
        """
        Initialize by saving a list of the raw tweets
        :param tweet_file: tweet file
        :param filter_by: the lexicon to filter by
        :param input_type: the input dataset source
        """
        self.matricies = self.initialize_matricies(filter_by)
        self.filter_by = filter_by
        self.input_type = input_type
        self.tweet_idx = self.get_tweet_idx()

        self.tweet_lines = [line for line in open(tweet_file, "r")]
        tweet_list = []
        # make sure all the lines are formatted in the expected way
        for line in self.tweet_lines:
            if self.tweet_idx < len(line.split("\t")):
                tweet_list.append(line.split("\t")[self.tweet_idx])
            else:
                self.tweet_lines.remove(line)
        self.tweet_list = tweet_list
        self.tag_list = CMUTweetTagger.runtagger_parse(self.tweet_list)

    def initialize_matricies(self, filter_by):
        """
        Initialize mapping matricies
        :param filter_by: the lexicon to filter by
        :return: mapping matricies
        """
        emo_list = []
        emo_matrix, w2idx, emo2idx = {}, {}, {}

        if filter_by == 'dm':
            emo_matrix = np.load('../lib/emotion_lexicon/DepecheMood/dm_emotion_lexicon_matrix.npy')
            w2idx = pickle.load(open('../lib/emotion_lexicon/DepecheMood/dm_word_map.pkl', "rb"))
            emo2idx = pickle.load(open('../lib/emotion_lexicon/DepecheMood/dm_emotion_map.pkl', "rb"))

        elif filter_by == 'nrc':
            emo_matrix = np.load('../lib/emotion_lexicon/NRC/nrc_emotion_lexicon_matrix.npy')
            w2idx = pickle.load(open('../lib/emotion_lexicon/NRC/nrc_word_map.pkl', "rb"))
            emo2idx = pickle.load(open('../lib/emotion_lexicon/NRC/nrc_emotion_map.pkl', "rb"))

        elif filter_by == 'kwsyns':
            emo_list = pickle.load(open('../lib/emotion_lexicon/emotion_kw_list/emotion_keywords_syns.pkl', "rb"))

        elif filter_by == 'kw':
            emo_list = pickle.load(open('../lib/emotion_lexicon/emotion_kw_list/emotion_keywords.pkl', "rb"))

        return emo_matrix, w2idx, emo2idx, emo_list

    def get_tweet_idx(self):
        """
        Get the index of the tweet given the source
        :return:
        """
        tweet_idx = 0

        if self.input_type == 'semeval16':
            tweet_idx = 3

        elif self.input_type in ('hashtag', 'semeval18'):
            tweet_idx = 1

        elif self.input_type == 'electoral':
            tweet_idx = 13

        return tweet_idx

    def filter_tweets(self):
        """
        Filter tweets that contain emotion words
        :return: string of tweets containing emotion words
        """
        tweet_list = []
        for index, line in enumerate(self.tweet_lines):
            if len(line.split("\t")) < self.tweet_idx:
                continue
            tweet = Tweet(line, self.tag_list[index], self.filter_by, self.input_type, self.matricies)
            if tweet.get_emotions():
                tweet_list.append(tweet.tweet_text.rstrip())
        return "\n".join(tweet_list)


def main():
    """
    Filter tweets by emotion lexicon
    :return: void
    """

    tweet_file = sys.argv[1]
    filter_by = sys.argv[2]
    input_type = sys.argv[3]
    out_name = sys.argv[4]

    output_file = '../data/preprocessed/filtered/' + filter_by + "_" + out_name + ".out"
    tok_file = '../data/preprocessed/tokenized/' + filter_by + "_" + out_name + ".tok"

    tf = TweetFile(tweet_file, filter_by, input_type)

    with open(tok_file, 'a+') as tok:
        for tags in tf.tag_list:
            tagged_toks = []
            for tag_tuple in tags:
                word = tag_tuple[0].encode('latin-1', 'ignore').decode('latin-1')
                tag = tag_tuple[1].encode('latin-1', 'ignore').decode('latin-1')
                tagged_toks.append("{}:{}".format(word, tag))
            print(" ".join(tagged_toks), file=tok)

    with open(output_file, 'a+') as out:
        print(tf.filter_tweets(), file=out)


if __name__ == "__main__":
    main()
