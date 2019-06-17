"""
Baseline rule-based system for emotion cause extraction of tweets
Based on outputs from the TweeboParser Dependency Parser
"""

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.baseline.dependency_tweet_loader import TweetLoader


class EmotionCauseRuleExtractor:
    """
    Apply rules to extract emotion causes from tweets
    """

    MODALS = ('may', 'might', 'could', 'should', 'would', 'will')
    PREP_CONJ = ('P', 'R', 'T')
    VERBS = ('V', 'L')

    def get_dependencies(self, word_list, deps):
        """
        Recursively gather all dependencies of the given nodes
        :param word_list: list of dependency nodes
        :param deps: current list of dependencies
        :return: dictionary of all dependencies
        """
        if not word_list:
            return deps

        else:
            word = word_list.pop()
            deps.append(word)
            word_list.extend(word.children)

        return self.get_dependencies(word_list, deps)

    def apply_rules(self, emo_word):
        """
        Apply the rules to get a cause for the given emotion in a tweet
        :param emo_word: the emotion word node in the dependency tree
        :return: the emotion and cause
        """
        cause = None
        if emo_word.pos in self.VERBS and emo_word.has_children():
            if emo_word.parent and emo_word.parent.text in self.MODALS and emo_word.parent.pos == 'V':
                # Apply Rule 2
                # Example: "The results may surprise you."
                cause = self.get_emotion_cause(emo_word.parent, 2)
            else:
                # Apply Rule 1
                # Example: "I love Bernie Sanders"
                # Example: "exhausted from putting groceries away"
                cause = self.get_emotion_cause(emo_word, 1)
        elif emo_word.pos == 'A':
            if emo_word.has_children():
                # Apply Rule 1
                # Example: "I'm so excited for the new episode of Hannibal tomorrow"
                cause = self.get_emotion_cause(emo_word, 1)
            elif emo_word.parent != 0 and emo_word.parent.pos in self.VERBS:
                # Apply Rule 3
                # Example: "You may be interested in this evening's BBC documentary"
                cause = self.get_emotion_cause(emo_word.parent, 3)

        if cause:
            return emo_word, cause
        else:
            return emo_word, None

    def get_emotion_cause(self, word, rule):
        """
        Get the emotion cause from the given dependency node and rule
        :param word: the root dependency node
        :param rule: the rule being applied
        :return: the emotion cause
        """
        deps = sorted(self.get_dependencies(word.children[:], []), key=lambda w: w.idx)
        lhs = [d for d in deps if d.idx < word.idx]
        rhs = [d for d in deps if d.idx > word.idx]

        if rule == 2:
            return self.strip_prepositions(lhs) if lhs else None
        elif rule == 3:
            return self.strip_prepositions(rhs[1:]) if len(rhs) > 2 else None
        else:
            return self.strip_prepositions(rhs) if rhs else None

    def strip_prepositions(self, phrase):
        """
        Strip prepositions, conjunctions from dependent phrase
        :param phrase: a list of dependency nodes representing a phrase
        :return:
        """
        if phrase[0].pos not in self.PREP_CONJ:
            return phrase

        return self.strip_prepositions(phrase[1:]) if phrase[1:] else None

    def build_emo_cause_list(self, emo_words, idx2tweets):
        """
        Build list of emotions and causes
        :param emo_words: list of emotion words for each tweet
        :param idx2tweets: mapping of tweets to index
        :return: sorted list of emotions and causes
        """
        emo_list = []
        for tweet_id, words in emo_words.items():
            for word in words:
                emo, cause = self.apply_rules(word)
                if cause:
                    emo_list.append((emo, cause, idx2tweets[tweet_id]))
        return sorted(emo_list)


def main():
    """
    Apply rules to Tweeboparser outputs and return emotion-cause relations when found for tweets
    :return: void
    """

    parsed_tweet_file = sys.argv[1]
    output = sys.argv[2]
    tweets = TweetLoader(parsed_tweet_file)
    tweets.extract_emo_relations()
    extractor = EmotionCauseRuleExtractor()
    emo_list = extractor.build_emo_cause_list(tweets.tweet2emo, tweets.idx2tweet)

    with open(output, 'w') as out:
        for emotion, cause, sentence in emo_list:
            cause = " ".join([d.text for d in cause])
            print("EMOTION: " + emotion.text + "\tCAUSE: " + cause + "\tTWEET: " + sentence, file=out)
            print("", file=out)


if __name__ == "__main__":
    main()
