"""
Filters tweets by verbs and adjectives in the emotion lexicon
"""


class Tweet:

    def __init__(self, tweet_line, tags, filter_by, input_type, matricies):
        """
        Initialize by saving the tweet and tags
        :param tweet_line: the given tweet
        :param tags: pos tags for the tweet
        :param filter_by: the lexicon to filter by
        :param input_type: the input dataset source
        :param matricies: pickled matricies
        """
        self.emo_matrix, self.w2idx, self.emo2idx, self.emo_list = matricies
        self.filter_by = filter_by
        self.tweet_line = tweet_line
        self.tagged = tags
        self.words = [t[0] for t in tags]
        self.tweet_text = self.source = self.emotions \
            = self.other_emotions = self.target \
            = self.emo_word = self.cause = None

        self.load_tweet_info(tweet_line, input_type)

    def load_tweet_info(self, tweet_line, input_type):
        """
        Load the info from the given tweet
        :param tweet_line: the tweet line from the data file
        :param input_type: the input dataset source
        :return: void
        """
        tweet_info = tweet_line.split("\t")
        if input_type in ('hashtag', 'semeval18'):
            self.tweet_text = tweet_info[1]
        elif input_type == 'semeval16':
            self.tweet_text = tweet_info[3]

    def get_emotion(self, idx):
        """
        Get the emotion from the index
        :param idx: index of the emotion in the mapping
        :return: the emotion string
        """
        return [e for e, i in self.emo2idx.items() if i == idx][0]

    def should_include(self, emo_idx_list, n, val):
        """
        Should this word be included?
        :param emo_idx_list: list of values for each emotion
        :param n: the emotion index
        :param val: the value for the given index
        :return: bool
        """
        max_val = max(emo_idx_list)
        exclude_list = []

        if self.filter_by == 'nrc':
            # exclude POSITIVE or NEGATIVE
            exclude_list = [5, 6]

        if self.filter_by == 'dm':
            # exclude emotions with confidence values under 50%
            if max_val < .5:
                return False

        return val == max_val and n not in exclude_list

    def get_emotions(self):
        """
        Get emotional words associated with this tweet
        :return:
        """
        word_emos = []
        for word in self.words:
            tag = [tag[1] for tag in self.tagged if tag[0] == word][0]

            # filter by keyword
            if self.filter_by in ('kwsyns', 'kw'):
                # V and A for hashtags, VB and JJ for semeval
                pos_list = ['V', 'A', 'VB', 'JJ']
                if word in self.emo_list and tag in pos_list:
                    word_emos.append(word)

            else:
                # depechemood lexicon has tagged words
                if self.filter_by == 'dm':
                    word = word + "#" + tag.lower()
                if word in self.w2idx.keys():
                    emo_idx_list = self.emo_matrix[self.w2idx[word]]
                    emo_list = [self.get_emotion(n) for n, val in enumerate(emo_idx_list)
                                if self.should_include(emo_idx_list, n, val)]

                    # V and A for hashtags, VB and JJ for semeval
                    pos_list = ['V', 'A', 'VB', 'JJ']
                    if emo_list and (tag in pos_list):
                        emo_list = word + ": " + ", ".join(emo_list)
                        word_emos.append(emo_list)
        return word_emos
