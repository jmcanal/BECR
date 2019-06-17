"""
Split the data into train, dev, devtest, and test
"""
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sklearn.model_selection import train_test_split
from lib.tweet_tagger import CMUTweetTagger


def get_splits(data):
    """
    Create all the splits
    :param data: the data to split
    :return:
    """
    train, test = train_test_split(data, test_size=0.3)
    train, dev = train_test_split(train, test_size=0.3)
    devtest, test = train_test_split(test, test_size=0.3)
    return train, dev, devtest, test


def print_tagged_sample(train_tok, train):
    """
    Print the training data with tokenized, POS tagged tweets for labeling
    :param train_tok: the output file for tagged tweets
    :param train: the training data
    :return:
    """
    tag_list = CMUTweetTagger.runtagger_parse(train)
    with open(train_tok, 'a+') as tok:
        for tags in tag_list:
            tagged_toks = []
            for tagged_tuple in tags:
                word = tagged_tuple[0].encode('utf-8').decode('latin1')
                tag = tagged_tuple[1].encode('utf-8').decode('latin1')
                tagged_toks.append("{}:{}".format(word, tag))
            print(" ".join(tagged_toks), file=tok)


def main():
    """
    Create train/test splits from tweet file
    :return: void
    """
    data_file = sys.argv[1]
    train_tok = sys.argv[2]
    train_file = sys.argv[3]
    dev_file = sys.argv[4]
    devtest_file = sys.argv[5]
    test_file = sys.argv[6]

    data = [line.strip("\n") for line in open(data_file, "r")]

    train, dev, devtest, test = get_splits(data)

    print_tagged_sample(train_tok, train)

    with open(train_file, 'w') as train_out:
        print("\n".join(train), file=train_out)

    with open(dev_file, 'w') as dev_out:
        print("\n".join(dev), file=dev_out)

    with open(devtest_file, 'w') as devtest_out:
        print("\n".join(devtest), file=devtest_out)

    with open(test_file, 'w') as test_out:
        print("\n".join(test), file=test_out)


if __name__ == "__main__":
    main()