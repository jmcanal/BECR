"""
Load dependency parsed tweets
"""
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pickle
from collections import defaultdict as dd
from src.baseline.word_node import WordNode as Word
from src.baseline.tweet import Tweet


class TweetLoader:

    POS_LIST = ('V', 'A')

    # For loading curated emotion keyword list
    emo_kws = pickle.load(open('../../lib/emotion_lexicon/emotion_kw_list/emotion_keywords.pkl', "rb"))

    idx2tweet = {}
    tweet2emo = dd(list)
    tweet_list = []

    def __init__(self, file_name):
        """
        Initialize this class by reading in the tweets
        :param file_name: the name of the input file
        """
        with open(file_name, mode='r', errors='ignore') as f:

            tweets = f.read()
            self.tweets = tweets.rstrip().split('\n\n')

    def extract_emo_relations(self):
        """
        Extract emotion words and dependency relations from tweets
        :return:
        """
        for tweet_idx, tweet in enumerate(self.tweets):
            tweet_tokens = []
            idx2word, child2parent = {}, {}
            for word in tweet.rstrip().split('\n'):
                if not word:
                    sys.stderr.write("wat")
                    continue
                curr_word = Word(word.rstrip().split('\t'), tweet_idx)
                idx2word[curr_word.idx] = curr_word
                child2parent[curr_word] = curr_word.parent

                # Isolate emotion words that are Verbs or Adjectives
                if curr_word.text in self.emo_kws and curr_word.pos in self.POS_LIST:
                    self.tweet2emo[tweet_idx].append(curr_word)
                    curr_word.is_emotion_word = True

                tweet_tokens.append(curr_word.text)

            # update tweet dictionary and add children to words
            self.add_relatives(child2parent, idx2word)
            tweet_text = " ".join(tweet_tokens)
            self.idx2tweet[tweet_idx] = tweet_text

            # Create Tweet object
            self.add_tweet(tweet_idx, tweet_text, tweet_tokens, list(idx2word.values()))

    def add_tweet(self, tweet_idx, tweet_text, tweet_tokens, words):
        """
        Create Tweet object
        :param tweet_idx: index of tweet
        :param tweet_text: raw text of Tweet
        :param tweet_tokens: tokenized list of words in Tweet
        :param words: list of Word objects
        :return:
        """
        this_tweet = Tweet(tweet_text)
        this_tweet.idx = tweet_idx
        this_tweet.tokens = tweet_tokens
        this_tweet.index = tweet_idx
        this_tweet.words = words
        this_tweet.emo_words = self.tweet2emo[tweet_idx]
        self.tweet_list.append(this_tweet)
        return this_tweet

    def add_relatives(self, child2parent, idx2word):
        """
        Add dependency relations to the nodes in the current mapping
        :param child2parent: mapping of child nodes to parent nodes
        :param idx2word: mapping of index to word
        :return: void
        """
        for child, parent in child2parent.items():
            if parent not in (0, -1):
                parent_word = idx2word[parent]
                parent_word.add_child(child)
                child.parent = parent_word
