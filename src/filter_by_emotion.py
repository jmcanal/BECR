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


class Tweet:

    def __init__(self, tweet_line, tags, filter_by, input_type, matricies):
        """
        Initialize by saving the tweet and tags
        :param tweet_line: the given tweet
        :param tags: pos tags for the tweet
        """
        self.emo_matrix, self.w2idx, self.emo2idx, self.emo_list = matricies
        self.filter_by = filter_by
        self.tweet_line = tweet_line
        self.tagged = tags
        self.words = [t[0] for t in tags]
        self.tweet_text = self.source = self.emotions \
            = self.other_emotions = self.target \
            = self.emo_word = self.cause = None

        self.load_tweet_info(tweet_line, input_type)

    def load_tweet_info(self, tweet_line, input_type):
        tweet_info = tweet_line.split("\t")
        if input_type == 'electoral':
            self.tweet_text = tweet_info[13]
            self.source = tweet_info[14]
            self.emotions = tweet_info[15].split(' or ')
            self.other_emotions = tweet_info[16]
            self.target = tweet_info[19]
            self.emo_word = tweet_info[20]
            self.cause = tweet_info[21]
        elif input_type in ('hashtag', 'semeval18'):
            self.tweet_text = tweet_info[1]
        elif input_type == 'semeval16':
            self.tweet_text = tweet_info[3]

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
        exclude_list = []

        if self.filter_by == 'nrc':
            # exclude POSITIVE or NEGATIVE
            exclude_list = [5, 6]

        if self.filter_by == 'dm':
            # exclude emotions with confidence values under 50%
            if max_val < .5:
                return False

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
            if self.filter_by in ('kwsyns', 'kw'):
                # V and A for hashtags, VB and JJ for semeval
                pos_list = ['V', 'A', 'VB', 'JJ']
                if word in self.emo_list and tag in pos_list:
                    word_emos.append(word)

            else:
                # depechemood lexicon has tagged words
                if self.filter_by == 'dm':
                    word = word + "#" + tag.lower()
                if word in self.w2idx.keys():
                    emo_idx_list = self.emo_matrix[self.w2idx[word]]
                    emo_list = [self.get_emotion(n) for n, val in enumerate(emo_idx_list)
                                if self.should_include(emo_idx_list, n, val)]

                    # V and A for hashtags, VB and JJ for semeval
                    pos_list = ['V', 'A', 'VB', 'JJ']
                    if emo_list and (tag in pos_list):
                        emo_list = word + ": " + ", ".join(emo_list)
                        word_emos.append(emo_list)
        return word_emos

class TweetFile:

    def __init__(self, tweet_file, filter_by, input_type):
        """
        Initialize by saving a list of the raw tweets
        :param tweet_file: tweet file
        """
        self.matricies = self.initialize_matricies(filter_by)
        self.filter_by = filter_by
        self.input_type = input_type
        self.tweet_idx = self.get_tweet_idx()

        self.tweet_lines = [line for line in open(tweet_file, "r")]
        tweet_list = []
        for line in self.tweet_lines:
            if self.tweet_idx < len(line.split("\t")):
                tweet_list.append(line.split("\t")[self.tweet_idx])
            else:
                print(tweet_file)
                self.tweet_lines.remove(line)
        self.tweet_list = tweet_list
        self.tag_list = CMUTweetTagger.runtagger_parse(self.tweet_list)

    def initialize_matricies(self, filter_by):
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
        tweet_idx = 0
        if self.input_type == 'semeval16':
            tweet_idx = 3
        elif self.input_type in ('hashtag', 'semeval18'):
            tweet_idx = 1
        elif self.input_type == 'electoral':
            tweet_idx = 13
        return tweet_idx

    def filter_tweets(self, debug_file):
        """
        Filter tweets that contain emotion words
        :return: string of tweets containing emotion words
        """
        tweet_list = []
        with open(debug_file, 'a+') as d:
            for index, line in enumerate(self.tweet_lines):
                if len(line.split("\t")) < self.tweet_idx:
                    print(debug_file)
                    continue
                if index > len(self.tag_list):
                    print("lines: " + str(len(self.tweet_lines)))
                    print("tweets: " + str(len(self.tweet_list)))
                    print("tags: " + str(len(self.tag_list)))
                tweet = Tweet(line, self.tag_list[index], self.filter_by, self.input_type, self.matricies)
                if tweet.get_emotions():
                    print("TWEET", tweet.tweet_text, file=d)
                    print("WORD(S):", file=d)
                    for item in tweet.get_emotions():
                        print(item, file=d)
                    if self.input_type == 'electoral':
                        # other info for labeled tweets
                        print("EMOTIONS", ", ".join(tweet.emotions), file=d)
                        print("OTHER EMOTION", tweet.other_emotions, file=d)
                        print("CLUE", tweet.emo_word, file=d)
                        print("SOURCE", tweet.source, file=d)
                        print("TARGET", tweet.target, file=d)
                        print("CAUSE", tweet.cause, file=d)
                    print("\n", file=d)
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

    print(tweet_file, file=sys.stderr)
    output_file = '../data/preprocessed/filtered/' + filter_by + "_" + out_name + ".out"
    debug_file = '../data/preprocessed/debug/' + filter_by + "_" + out_name + ".txt"
    tok_file = '../data/preprocessed/tokenized/' + filter_by + "_" + out_name + ".tok"

    tf = TweetFile(tweet_file, filter_by, input_type)

    with open(tok_file, 'a+') as tok:
        for tags in tf.tag_list:
            tagged_toks = []
            for tuple in tags:
                word = tuple[0].encode('latin-1', 'ignore').decode('latin-1')
                tag = tuple[1].encode('latin-1', 'ignore').decode('latin-1')
                tagged_toks.append("{}:{}".format(word, tag))
            print(" ".join(tagged_toks), file=tok)

    with open(output_file, 'a+') as out:
        print(tf.filter_tweets(debug_file), file=out)

if __name__ == "__main__":
    main()
