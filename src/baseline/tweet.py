
class Tweet:
    """
    Tweet object for baseline algorithm
    """

    words = []
    tokens = []
    emo_words = []

    mapping = {}
    index = None
    seed = None

    def __init__(self, raw):
        """
        Initialize by saving the raw tweet
        :param raw:
        """
        self.raw = raw

    def __str__(self):
        """
        Return the raw tweet string
        :return:
        """
        return self.raw
