"""
Evaluate the emotion cause extraction system
"""

import os
import sys
import random
from difflib import SequenceMatcher


class EmoCause:

    def __init__(self, tweet, emo, cause, relaxed):
        self.tweet = tweet
        self.emo = emo
        self.cause = cause
        self.percent = 0.99 if relaxed else 0.7

    def __eq__(self, other):
        t_sim = SequenceMatcher(a=self.tweet, b=other.tweet).ratio()
        tweet_eq = t_sim > self.percent
        e_sim = SequenceMatcher(a=self.emo, b=other.emo).ratio()
        emo_eq = e_sim > self.percent
        c_sim = SequenceMatcher(a=self.cause, b=other.cause).ratio()
        cause_eq = c_sim > self.percent
        return tweet_eq and emo_eq and cause_eq

    def __hash__(self):
        return hash(self.emo + self.cause)


def load_emo_causes(emotion_cause_file, relaxed=False):
    """
    Load emotion cause results into a list of tuples
    :param emotion_cause_file: an emotion cause file
    :return: list of tuples
    """
    with open(emotion_cause_file, mode='r', errors='ignore') as ec:
        tweets = ec.read()
        tweets = tweets.split('\n\n')

    emotion_cause_list = set()
    for tweet in tweets:
        if not tweet:
            continue
        tweet_info = tweet.split('\t')
        tweet_text = tweet_info[2].split(':')[1].strip()
        emotion = tweet_info[0].split(':')[1].strip()
        cause = tweet_info[1].split(':')[1].strip()

        emotion_cause_list.add(EmoCause(tweet_text, emotion, cause, relaxed))

    return emotion_cause_list

def calculate_recall(cause_file, gold_file, total, relaxed=False):
    """
    Calculate recall based on the randomly sampled labeled file
    :param cause_file: the emotion cause output file
    :param gold_file: the labeled file
    :return: the recall value
    """
    gold_labels = load_emo_causes(gold_file, relaxed)
    emo_causes = load_emo_causes(cause_file, relaxed)

    missing = len(gold_labels.intersection(emo_causes))

    return (25. - float(missing)) / 25.

def pull_precision(cause_file, precision_file):
    """
    Randomly sample 20 tweets to label for precision
    :param cause_file: the emotion cause output file
    :param precision_file: the labeled file
    :return: void
    """
    emo_causes, emo_cause_dict = load_emo_causes(cause_file)
    sample_emo_causes = random.sample(emo_causes, 25)

    with open(precision_file, 'w') as p:
        for emo_cause in sample_emo_causes:
            print("EMOTION: " + emo_cause[1] + "\tCAUSE: " + emo_cause[2] + '\tTWEET: ' + emo_cause[0], file=p)

def get_precision(precision_file, total, relaxed=False):
    """
    Calculate precision from the labeled precision file
    :param precision_file: the labeled precision file
    :return: the precision value
    """
    with open(precision_file, 'r', errors='ignore') as p:
        tweets = p.read().split('\n')

    correct = 0
    for tweet in tweets:
        if tweet.startswith('1'):
            correct += 1
        if relaxed and tweet.startswith('0.5'):
            correct += 1

    return float(correct) / float(total)

def calculate_f_score(precision, recall):
    """
    Calculate F score from the precision and recall values
    :param precision: the precision value
    :param recall: the recall value
    :return: the f-score value
    """
    return (2. * precision * recall) / (precision + recall)

def main():
    """
    Calculation evaluation metrics
    :return: void
    """
    emotion_cause_file = sys.argv[1]
    gold_label_file = sys.argv[2]
    precision_file = sys.argv[3]
    output_file = sys.argv[4]
    total = sys.argv[5]

    with open(output_file, 'w') as out:
        recall = calculate_recall(emotion_cause_file, gold_label_file, total)
        print("Recall: " + str(recall), file=out)
        if os.stat(precision_file).st_size == 0:
            pull_precision(emotion_cause_file, precision_file)
        else:
            precision = get_precision(precision_file, total)
            f_score = calculate_f_score(precision, recall)
            print("Precision: " + str(precision), file=out)
            print("F-Score: " + str(f_score), file=out)

    with open(output_file + '_relaxed', 'w') as out:
        recall = calculate_recall(emotion_cause_file, gold_label_file, total, True)
        print("Recall: " + str(recall), file=out)
        if os.stat(precision_file).st_size == 0:
            pull_precision(emotion_cause_file, precision_file)
        else:
            precision = get_precision(precision_file, total, True)
            f_score = calculate_f_score(precision, recall)
            print("Precision: " + str(precision), file=out)
            print("F-Score: " + str(f_score), file=out)


if __name__ == "__main__":
    main()