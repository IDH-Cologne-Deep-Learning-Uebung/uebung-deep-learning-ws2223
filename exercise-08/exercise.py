import bz2

import numpy as np

from keras import models, layers
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import pad_sequences
from keras import regularizers


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

print("Running...")
MAX_LENGTH = 5

train_labels, train_texts = get_labels_and_texts('data/train.ft.txt.bz2')
print("Training data retrieved.")
test_labels, test_texts = get_labels_and_texts('data/test.ft.txt.bz2')
print("Test data retrieved.")
tokenizer = Tokenizer()
print("Tokenizer initialized.")
train_sequences = tokenizer.texts_to_sequences(train_texts)
train_array = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding="post")
print("Training data prepared.")
test_sequences = tokenizer.texts_to_sequences(test_texts)
print("Test data prepared.")
test_array = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding="post")

ffnn = models.Sequential()
print("Model initialized.")
ffnn.add(layers.Input(shape=(MAX_LENGTH,)))
print("Input layer added.")
ffnn.add(layers.Embedding(MAX_LENGTH, embeddings_regularizer=regularizers.L2(0.01), output_dim=100))
ffnn.add(layers.Flatten())
print("Embedding layer added.")
ffnn.add(layers.Dense(100, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
print("Dense layer added.")
ffnn.add(layers.Dense(50, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
print("Dense layer added.")
ffnn.add(layers.Dense(1, activation="sigmoid"))
print("Output layer added.")

print("\nSummary:")
ffnn.summary()
print()

ffnn.compile(loss="binary_crossentropy", optimizer="sgd",
  metrics=["accuracy"])
print("Model compiled.")

ffnn.fit(train_array, train_labels, epochs=10, batch_size=10, verbose=1)
print("Model trained.")

ffnn.evaluate(test_array, test_labels)
