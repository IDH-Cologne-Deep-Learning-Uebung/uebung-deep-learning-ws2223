import bz2

import numpy as np

from tensorflow.python.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def get_labels_and_texts(file, n=12000):
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
test_labels, test_texts = get_labels_and_texts('data/test.ft.txt.bz2')

variable = get_labels_and_texts('data/test.ft.txt.bz2')
test_labels = variable[0]
test_texts = variable[1]


vectorizer = CountVectorizer(max_features=5000)
vectorizer.fit(train_texts)

train_texts_vec = vectorizer.transform(train_texts)
test_texts_vec = vectorizer.transform(test_texts)

x_train = train_texts_vec
y_train = train_labels

x_test = test_texts_vec
y_test = test_labels



################
# Base variant #
################

model = models.Sequential()
model.add(layers.Dense(100, input_shape=(5000,), activation="sigmoid"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="sgd",
  metrics=["accuracy"])

model.fit(x_train, y_train, epochs=25, batch_size=5, verbose=0)

print("Base variant")
model.evaluate(x_test, y_test)



###################
# Dropout variant #
###################

model = models.Sequential()
model.add(layers.Dense(100, input_shape=(5000,), activation="sigmoid"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="sgd",
  metrics=["accuracy"])

model.fit(x_train, y_train, epochs=25, batch_size=5, verbose=0)

print("Dropout variant")
model.evaluate(x_test, y_test)



##########################
# Regularization variant #
##########################

model = models.Sequential()
model.add(layers.Input(shape=(5000,)))
model.add(layers.Dense(100, activation="sigmoid", kernel_regularizer=regularizers.l2(0.2)))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="sgd",
  metrics=["accuracy"])

model.fit(x_train, y_train, epochs=25, batch_size=5, verbose=0)

print("Regularization variant")
model.evaluate(x_test, y_test)
