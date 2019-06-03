"""
Load dependency parsed tweets for SnowBreds algorithm
"""
import sys
import pickle
from collections import defaultdict as dd
from src.word_node import WordNode as Word
from src.tweet import Tweet
from src.baseline.dependency_tweet_loader import TweetLoader


class SnowBredsTweetLoader(TweetLoader):

    NEGATION = ('don\'t', 'doesn\'t', 'didn\'t', 'won\'t', 'not', 'no', 'isn\'t')
    NEGATED_POS = ("R", "D", "V")
    ADVERB = "R"
    VERB = "V"

    def __init__(self, file):
        self.loader = TweetLoader(file)  #todo: switch to super method?

    def extract_emo_relations(self):
        """
        Extract emotion words and dependency relations from tweets
        Modified from Baseline TweetLoader to store emotions as lists
        of Word objects instead of single Word objects
        :return:
        """
        for tweet_idx, tweet in enumerate(self.loader.tweets):
            tweet_tokens = []
            idx2word, child2parent = {}, {}
            for word in tweet.rstrip().split('\n'):
                if not word:
                    sys.stderr.write("wat")
                    continue
                curr_word = Word(word.rstrip().split('\t'), tweet_idx)
                idx2word[curr_word.idx] = curr_word
                child2parent[curr_word] = curr_word.parent

                tweet_tokens.append(curr_word.original_text)

            # update tweet dictionary and add children to words
            self.loader.add_relatives(child2parent, idx2word)
            tweet_text = " ".join(tweet_tokens)
            self.loader.idx2tweet[tweet_idx] = tweet_text

            # Create Tweet object
            tweet = self.loader.add_tweet(tweet_idx, tweet_text, tweet_tokens, list(idx2word.values()))

            # Isolate emotion words that are Verbs or Adjectives
            for word in tweet.words:
                if word.text in self.loader.emo_kws and word.pos in self.loader.POS_LIST:
                    word_context = self.get_word_context(word, tweet)
                    self.loader.tweet2emo[tweet_idx].append(word)
                    word.emo = True
                    word.phrase = word_context

    def get_word_context(self, word, tweet):
        """
        Get full word context up to two preceding words, capturing negation todo: and emphasis, e.g. "so"? TBD
        :param word: Word object, an emotion word
        :param tweet: Tweet object corresponding to emo word
        :return:
        """
        word_context = [word]
        prev = tweet.words[word.idx - 2] if word.idx > 0 else None
        prev_prev = tweet.words[word.idx - 3] if word.idx > 1 else None

        # Not currently capturing longer-distance negation,
        # e.g. "I don't think he has to worry about turning into Batman"
        if prev:
            # Looking for examples like "not afraid"
            if prev.text in self.NEGATION:
                word_context.insert(0, prev)
            elif prev_prev:
                # Looking for examples like: "Don't really like" or "Won't be happy"
                if prev_prev.text in self.NEGATION and (prev.pos in self.ADVERB or prev.pos in self.VERB): #todo check this because maybe it's too broad
                    word_context = [prev_prev, prev] + word_context

        return word_context