"""
Evaluate the emotion cause extraction system
"""

import os
import sys
import random
from difflib import SequenceMatcher

def load_emo_causes(emotion_cause_file):
    """
    Load emotion cause results into a list of tuples
    :param emotion_cause_file: an emotion cause file
    :return: list of tuples
    """
    with open(emotion_cause_file, 'r') as ec:
        tweets = ec.read().split('\n')

    emotion_cause_list = set()
    emotion_cause_dict = {}
    for tweet in tweets:
        if not tweet:
            continue
        tweet_info = tweet.split('\t')
        tweet_text = tweet_info[2].split(': ')[1]
        emotion = tweet_info[0].split(': ')[1]
        cause = tweet_info[1].split(': ')[1]

        emotion_cause_list.add((tweet_text, emotion, cause))
        emotion_cause_dict[tweet_text] = (emotion, cause)

    return emotion_cause_list, emotion_cause_dict

def calculate_recall(cause_file, gold_file, out):
    """
    Calculate recall based on the randomly sampled labeled file
    :param cause_file: the emotion cause output file
    :param gold_file: the labeled file
    :return: the recall value
    """
    gold_labels, gold_tweet_dict = load_emo_causes(gold_file)
    emo_causes, emo_tweet_dict = load_emo_causes(cause_file)
    count = 0

    print("Recall:", file=out)

    for tweet1, emo_cause1 in gold_tweet_dict.items():
        count += 1
        gold_tweet = tweet1.lower()
        print("\n{}. ".format(count) + tweet1, file=out)
        print(str(emo_cause1), file=out)
        for tweet2, emo_cause2 in emo_tweet_dict.items():
            if SequenceMatcher(a=gold_tweet, b=tweet2).ratio() > 0.99:
                print("BASELINE" + str(emo_cause2), file=out)

    # missing = len(gold_labels - emo_causes)

    # return (25. - float(missing)) / 25.

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

def get_precision(precision_file):
    """
    Calculate precision from the labeled precision file
    :param precision_file: the labeled precision file
    :return: the precision value
    """
    with open(precision_file, 'r') as p:
        tweets = p.read().split('\n')

    correct = 0
    for tweet in tweets:
        if tweet.startswith('1'):
            correct += 1

    return float(correct) / 20.

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

    with open(output_file, 'w') as out:
        calculate_recall(emotion_cause_file, gold_label_file, out)
        # print("Recall: ", file=out)
        if os.stat(precision_file).st_size == 0:
            pull_precision(emotion_cause_file, precision_file)
        # else:
        #     precision = get_precision(precision_file)
        #     f_score = calculate_f_score(precision, recall)
        #     print("Precision: " + str(precision), file=out)
        #     print("F-Score: " + str(f_score), file=out)

if __name__ == "__main__":
    main()