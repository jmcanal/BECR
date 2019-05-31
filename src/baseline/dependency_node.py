"""
A node of the dependency parse
"""
import sys
class DependencyNode:

    POS_LIST = ('MW', 'CONJ')

    def __init__(self, word_feats, tweet):
        """
        Initialize this node with the word and features
        :param word_feats: list of word features
        :param tweet: the tweet this node is in
        """
        self.idx = int(word_feats[0])
        self.text = word_feats[1]
        self.pos = word_feats[3]
        self.parent = int(word_feats[6])
        self.mw = word_feats[7] if word_feats[7] in self.POS_LIST else None
        self.tweet_idx = tweet
        self.children = []

    def add_child(self, child):
        """
        Add a child to this node
        :param child: the child to add
        :return: void
        """
        self.children.append(child)

    def has_children(self):
        """
        Does this node have children?
        :return: bool
        """
        return len(self.children) > 0

    def __str__(self):
        """
        Represent this node as its text
        :return: str
        """
        return self.text

    def get_children(self):
        """
        Get the children of this node as a string
        :return: str
        """
        return ", ".join([w.text for w in self.children])
    
    def __lt__(self, other):
        """
        Words are ordered by their alphanumeric order
        :param other: another object
        :return: int
        """
        return self.text < other.text
