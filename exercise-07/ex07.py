import bz2
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras import layers, Sequential, regularizers

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

# Aufgabe 3

vectorizer = CountVectorizer()
y_train, x_train = get_labels_and_texts("data/train.ft.txt.bz2")
y_test, x_test = get_labels_and_texts("data/test.ft.txt.bz2")
vectorizer.fit(x_train)

vector_train = vectorizer.transform(x_train)
vector_test = vectorizer.transform(x_test)


# Aufgabe 4

model = Sequential()
model.add(layers.Input(shape=(34106,)))
model.add(layers.Dense(15, activation="sigmoid"))
model.add(layers.Dense(1, activation="sigmoid", activity_regularizer=regularizers.L2(0.2)))
model.add(layers.Dropout(0.5))

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(vector_train, y_train, epochs=10, batch_size=5)

model.evaluate(vector_test, y_test)
