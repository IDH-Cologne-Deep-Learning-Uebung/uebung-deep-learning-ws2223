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
# print("Building the NN...")
# model = keras.Sequential()
# model.add(layers.Input(shape=(1000)))
# model.add(layers.Dense(10, activation="softmax"))
# model.add(layers.Dense(10, activation="softmax"))
# model.add(layers.Dense(1, activation="softmax"))
# print("Compiling...")
# model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

# print("Training the NN...")
# model.fit(X_train, y_train, epochs=5, batch_size=1)
# y_pred = model.predict(X_test)
# print('Precision: ', precision_score(y_pred=y_pred, y_true=y_test))
# print('Recall: ', recall_score(y_pred=y_pred, y_true=y_test))
# print('F1: ', f1_score(y_pred=y_pred, y_true=y_test))

# Model with regularization
# print("Building the NN...")
# model = keras.Sequential()
# model.add(layers.Input(shape=(1000)))
# model.add(layers.Dense(10, activation="softmax", activity_regularizer=regularizers.L2(0.2)))
# model.add(layers.Dense(10, activation="softmax", activity_regularizer=regularizers.L2(0.2)))
# model.add(layers.Dense(1, activation="softmax", activity_regularizer=regularizers.L2(0.2)))
# print("Compiling...")
# model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

# Model with dropout
print("Building the NN...")
model = keras.Sequential()
model.add(layers.Input(shape=(1000)))
model.add(layers.Dense(10, activation="softmax"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation="softmax"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation="softmax"))
print("Compiling...")
model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

print("Training the NN...")
model.fit(X_train, y_train, epochs=15, batch_size=30)
y_pred = model.predict(X_test)
print('Precision: ', precision_score(y_pred=y_pred, y_true=y_test))
print('Recall: ', recall_score(y_pred=y_pred, y_true=y_test))
print('F1: ', f1_score(y_pred=y_pred, y_true=y_test))