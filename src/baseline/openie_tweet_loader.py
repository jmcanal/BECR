"""
Load openie parsed tweets
"""
from nltk import pos_tag


class TweetLoader:
    """
    Processes OpenIE outputs
    - Converts triples to format that can be matched to predefined emotion-cause patterns
    - Includes preprocessing steps of adding POS tags and emotion values
    """

    patterns = []
    tweets = []

    def __init__(self, relation_file):
        """
        Initialize the class by splitting the OpenIE file into tweet descriptor sections
        :param relation_file: the OpenIE output file with relations extracted from tweets
        """
        with open(relation_file, 'r') as f:
            self.tweet_relations = f.read().split('\n\n')
        self.get_patterns()

    def get_patterns(self):
        """
        Filter the output relations to find patterns we want to include
        :return: the list of patterns and the raw tweets
        """
        for idx, tweet in enumerate(self.tweet_relations):

            tweet_info = tweet.split('\n')
            self.tweets.append(tweet_info[0])

            for relation in tweet_info[1:]:
                triple = relation.split(" ", 1)[1].strip('()\n')

                # context is not a relation that we care about
                if triple.startswith('Context'):
                    continue

                triple = [w for w in triple.split('; ') if w]
                if len(triple) > 2:
                    self.patterns.append((idx, pos_tag(triple)))
