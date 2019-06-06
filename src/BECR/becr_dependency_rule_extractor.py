"""
Baseline rule-based system for emotion cause extraction of tweets
Based on outputs from the TweeboParser Dependency Parser
"""

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.baseline.dependency_rule_extractor import EmotionCauseRuleExtractor


class BECREmotionCauseRuleExtractor(EmotionCauseRuleExtractor):
    """
    Apply rules to extract emotion causes from tweets
    Expanded from Base Dependency Rule Extractor to include additional candidate emo-cause relation pairs
    """

    def apply_rules(self, emo_word):
        """
        Function to grab as examples of emotions and potential causes
        Acts as pre-processing step to create a set of emo-cause pairs to search first for seed examples and then to
        find extensions
        :param emo_word: the emotion word (emo_context[-1]
        :return: list of emotion cause tuples
        """
        deps1, deps2 = [], []
        if emo_word.pos in self.VERBS and emo_word.has_children():
            if emo_word.parent and emo_word.parent.text in self.MODALS and emo_word.parent.pos == 'V':
                # Apply Rule 2
                # Example: "The results may surprise you."
                deps1 = self.get_emotion_cause(emo_word.parent, 2)

                # Also apply Rule 1
                # Example: "You may love george harrison"
                deps2 = self.get_emotion_cause(emo_word, 1)

            else:
                # Apply Rule 1
                # Example: "I love Bernie Sanders"
                # Example: "exhausted from putting groceries away"
                deps1 = self.get_emotion_cause(emo_word, 1)

                # Also apply Rule 2
                # Example: "her tweets surprise me"
                # Example: "why doesn't that surprise me"
                deps2 = self.get_emotion_cause(emo_word, 2)

        elif emo_word.pos == 'A':

            if emo_word.has_children():
                # Apply Rule 1
                # Example: "I'm so excited for the new episode of Hannibal tomorrow"
                deps1 = self.get_emotion_cause(emo_word, 1)

            elif emo_word.parent != 0:
                # Apply Rule 3
                # Example: "You may be interested in this evening's BBC documentary"
                if emo_word.parent.pos in self.VERBS:
                    deps1 = self.get_emotion_cause(emo_word.parent, 3)

        else:
            return []

        if deps2:
            return [(emo_word, deps1), (emo_word, deps2)]
        elif deps1:
            return [(emo_word, deps1)]
        else:
            return []

    def build_emo_cause_list(self, emo_words, idx2tweets):
        """
        Build list of emotions and causes with more candidates than baseline
        :param emo_words: list of emotion words for each tweet
        :param idx2tweets: mapping of tweets to index
        :return: sorted list of emotions and causes
        """
        emo_lookup = []
        for sent_id, words in emo_words.items():
            for word in words:
                emo_causes = self.apply_rules(word)
                for emo, cause in emo_causes:
                    if cause:
                        emo_lookup.append([emo, cause, sent_id])
        return emo_lookup
