import bz2
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import keras
from keras import layers, regularizers
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


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

vectorizer = CountVectorizer(
    max_features=1000,
    lowercase=True
)

print("Reading in the files...")
y_train, X_train = get_labels_and_texts("data/train.ft.txt.bz2")
y_test, X_test = get_labels_and_texts("data/test.ft.txt.bz2")

print("Fitting and transforming the text data...")
vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

# Model without regularization or dropout
model_1 = keras.Sequential()
model_1.add(layers.Input(shape=(1000)))
model_1.add(layers.Dense(10, activation="softmax"))
model_1.add(layers.Dense(10, activation="softmax"))
model_1.add(layers.Dense(1, activation="softmax"))
model_1.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

model_1.fit(X_train, y_train, epochs=5, batch_size=1, verbose=0)
model_1.evaluate(X_train, y_test)

# Model with regularization
model_2 = keras.Sequential()
model_2.add(layers.Input(shape=(1000)))
model_2.add(layers.Dense(10, activation="softmax", activity_regularizer=regularizers.L2(0.2)))
model_2.add(layers.Dense(10, activation="softmax", activity_regularizer=regularizers.L2(0.2)))
model_2.add(layers.Dense(1, activation="softmax", activity_regularizer=regularizers.L2(0.2)))
model_2.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

model_2.fit(X_train, y_train, epochs=5, batch_size=1, verbose=0)
model_2.evaluate(X_train, y_test)

# Model with dropout
model_3 = keras.Sequential()
model_3.add(layers.Input(shape=(1000)))
model_3.add(layers.Dense(10, activation="softmax"))
model_3.add(layers.Dropout(0.2))
model_3.add(layers.Dense(10, activation="softmax"))
model_3.add(layers.Dropout(0.5))
model_3.add(layers.Dense(1, activation="softmax"))
model_3.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

model_3.fit(X_train, y_train, epochs=25, batch_size=5, verbose=0)
model_3.evaluate(X_train, y_test)