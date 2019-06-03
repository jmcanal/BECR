class Tweet:

    def __init__(self, raw):
        self.raw = raw
        self.words = []
        self.tokens = []
        self.mapping = {}
        self.emo_words = []
        self.index = None
        self.seed = None

    def __str__(self):
        return self.raw