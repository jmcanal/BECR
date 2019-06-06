"""
Load dependency parsed tweets for BECR algorithm
"""
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.baseline.word_node import WordNode as Word
from src.baseline.dependency_tweet_loader import TweetLoader


class BECRTweetLoader(TweetLoader):

    NEGATION = ('don\'t', 'doesn\'t', 'didn\'t', 'won\'t', 'not', 'no', 'isn\'t')
    NEGATED_POS = ("R", "D", "V")
    ADVERB = "R"
    VERB = "V"

    def extract_emo_relations(self):
        """
        Extract emotion words and dependency relations from tweets
        Modified from Baseline TweetLoader to store emotions as lists
        of Word objects instead of single Word objects
        :return: void
        """
        for tweet_idx, tweet in enumerate(self.tweets):
            tweet_tokens = []
            idx2word, child2parent = {}, {}
            for word in tweet.rstrip().split('\n'):
                curr_word = Word(word.rstrip().split('\t'), tweet_idx)
                idx2word[curr_word.idx] = curr_word
                child2parent[curr_word] = curr_word.parent

                tweet_tokens.append(curr_word.original_text)

            # update tweet dictionary and add children to words
            self.add_relatives(child2parent, idx2word)
            tweet_text = " ".join(tweet_tokens)
            self.idx2tweet[tweet_idx] = tweet_text

            # Create Tweet object
            tweet = self.add_tweet(tweet_idx, tweet_text, tweet_tokens, list(idx2word.values()))

            # Isolate emotion words that are Verbs or Adjectives
            for word in tweet.words:
                if word.text in self.emo_kws and word.pos in self.POS_LIST:
                    word_context = self.get_word_context(word, tweet)
                    self.tweet2emo[tweet_idx].append(word)
                    word.emotion = True
                    word.phrase = word_context

    def get_word_context(self, word, tweet):
        """
        Get full word context up to two preceding words, capturing negation
        :param word: Word object, an emotion word
        :param tweet: Tweet object corresponding to emo word
        :return: list of context words
        """
        word_context = [word]
        prev = tweet.words[word.idx - 2] if word.idx > 0 else None
        prev_prev = tweet.words[word.idx - 3] if word.idx > 1 else None

        # Not currently capturing longer-distance negation, or lots of modals
        # e.g. "I would have been happy if we went there" or "I don't usually really like sushi"
        if prev:
            # looking for examples like: "not afraid" or "so excited"
            if prev.text in self.NEGATION or prev.pos in self.ADVERB:
                word_context.insert(0, prev)
            if prev_prev:
                # Looking for examples like: "Don't really like," "Won't be happy," or "so very happy"
                if prev_prev.text in self.NEGATION and (prev.pos in self.ADVERB or prev.pos in self.VERB):
                    word_context = [prev_prev, prev] + [word]

        return word_context
