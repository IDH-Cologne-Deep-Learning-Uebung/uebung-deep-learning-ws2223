import bz2

import numpy as np

from tensorflow.python.keras import models, layers
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers


def get_labels_and_texts(file, n=10000):
    labels = []
    texts = []
    i = 0
    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        labels.append(int(x[9]) - 1)
        texts.append(x[10:].strip())
        i = i + 1
        if i >= n:
          return np.array(labels), texts
    return np.array(labels), texts


train_labels, train_texts = get_labels_and_texts('data/train.ft.txt.bz2')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1
train_texts = tokenizer.texts_to_sequences(train_texts)

MAX_LENGTH = max(len(train_ex) for train_ex in train_texts)

train_texts = pad_sequences(train_texts, maxlen=MAX_LENGTH, padding="post")

ffnn = models.Sequential()
ffnn.add(layers.Input(shape=(MAX_LENGTH)))
ffnn.add(layers.Embedding(vocab_size, 200, input_length=MAX_LENGTH))
ffnn.add(layers.Flatten())
ffnn.add(layers.Dense(100, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(50, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(1, activation="sigmoid"))

ffnn.summary()

ffnn.compile(loss="binary_crossentropy", optimizer="sgd",
  metrics=["accuracy"])

ffnn.fit(train_texts, train_labels, epochs=10, batch_size=10, verbose=1)


