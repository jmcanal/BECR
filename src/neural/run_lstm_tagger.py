import sys
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
from keras import backend as K


class Tagger:

    def __init__(self, train_word_file, train_tag_file, test_word_file, test_tag_file):
        self. train_sentences = self.read_file(train_word_file)
        self. train_tags = self.read_file(train_tag_file)
        self. test_sentences = self.read_file(test_word_file)
        self. test_tags = self.read_file(test_tag_file)

        self.w2idx = self.build_word_mapping(self.train_sentences)
        self.t2idx = self.build_tag_mapping(self.train_tags)

        self.MAX_LENGTH = len(max(self.train_sentences, key=len))

        self.train_sentence_vecs = self.create_word_vectors(self.train_sentences)
        self.train_tag_vecs = self.create_tag_vectors(self.train_tags)
        self.test_sentence_vecs = self.create_word_vectors(self.test_sentences)
        self.test_tag_vecs = self.create_tag_vectors(self.test_tags)


        self.model = self.setup_model()

    def read_file(self, file_name):
        with open(file_name, 'r', errors='ignore') as f:
            array = [sentence.split() for sentence in f]

        return array

    def build_word_mapping(self, sentences):
        words = set([])

        for s in sentences:
            for w in s:
                words.add(w.lower())

        word2index = {w: i + 2 for i, w in enumerate(list(words))}
        word2index['-PAD-'] = 0  # The special value used for padding
        word2index['-OOV-'] = 1  # The special value used for OOVs

        return word2index

    def build_tag_mapping(self, tag_lines):
        tags = set([])

        for ts in tag_lines:
            for t in ts:
                tags.add(t)

        tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
        tag2index['-PAD-'] = 0  # The special value used to padding

        return tag2index

    def create_word_vectors(self, sentences):
        sentence_vecs = []
        for s in sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(self.w2idx[w.lower()])
                except KeyError:
                    s_int.append(self.w2idx['-OOV-'])

            sentence_vecs.append(s_int)

        return pad_sequences(sentence_vecs, maxlen=self.MAX_LENGTH, padding='post')

    def create_tag_vectors(self, tags):
        tag_vecs = []
        for s in tags:
            tag_vecs.append([self.t2idx[t] for t in s])

        return pad_sequences(tag_vecs, maxlen=self.MAX_LENGTH, padding='post')

    def setup_model(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(self.MAX_LENGTH,)))
        model.add(Embedding(len(self.w2idx), 128))
        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(TimeDistributed(Dense(len(self.t2idx))))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(0.001),
                      metrics=['accuracy', self.ignore_class_accuracy(0)])

        return model

    def make_one_hot(self, sequences, categories):
        cat_sequences = []
        for s in sequences:
            cats = []
            for item in s:
                cats.append(np.zeros(categories))
                cats[-1][item] = 1.0
            cat_sequences.append(cats)
        return np.array(cat_sequences)

    def train_model(self):
        tag_one_hot = self.make_one_hot(self.train_tag_vecs, len(self.t2idx))
        self.model.fit(self.train_sentence_vecs, tag_one_hot, batch_size=128, epochs=40, validation_split=0.2)

    def ignore_class_accuracy(self, to_ignore=0):
        def ignore_accuracy(y_true, y_pred):
            y_true_class = K.argmax(y_true, axis=-1)
            y_pred_class = K.argmax(y_pred, axis=-1)

            ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
            matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
            accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
            return accuracy

        return ignore_accuracy

    def un_one_hot(self, sequences, index):
        token_sequences = []
        for categorical_sequence in sequences:
            token_sequence = []
            for categorical in categorical_sequence:
                token_sequence.append(index[np.argmax(categorical)])

            token_sequences.append(token_sequence)

        return token_sequences

    def evaluate(self):
        scores = self.model.evaluate(self.test_sentence_vecs, self.make_one_hot(self.test_tag_vecs, len(self.t2idx)))
        return f"{self.model.metrics_names[1]}: {scores[1] * 100}"  # acc: 99.09751977804825

    def predict(self, test_sentences):
        sentence_vectors = self.create_word_vectors(test_sentences)
        predictions = self.model.predict(sentence_vectors)
        return self.un_one_hot(predictions, {i: t for t, i in self.t2idx.items()})


def main():
    """
    Create emotion lexicon matricies and lists from lexicon
    :return:
    """
    train_sentences = sys.argv[1]
    train_tags = sys.argv[2]
    test_sentences = sys.argv[3]
    test_tags = sys.argv[4]
    acc_file = sys.argv[5]
    pred_file = sys.argv[6]

    tagger = Tagger(train_sentences, train_tags, test_sentences, test_tags)
    tagger.train_model()

    with open(acc_file, 'w') as acc:
        print(tagger.evaluate(), file=acc)

    with open(pred_file, 'w') as pred:
        for s, predicted, real in zip(tagger.test_sentences, tagger.predict(tagger.test_sentences), tagger.test_tags):
            print("sentence: " + " ".join(s), file=pred)
            print("prediction: " + " ".join(predicted), file=pred)
            print("gold: " + " ".join(real), file=pred)


if __name__ == "__main__":
    main()