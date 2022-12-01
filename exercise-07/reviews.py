import bz2
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential

# Step 2: Download data
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

# print(get_labels_and_texts("data/train.ft.txt.bz2", 4)[1])

# Step 3: Represent text
# vok wert: 5000

vectorizer = CountVectorizer(max_features = 2, lowercase=True) # Parameter vocabulary = 5000 spuckt Fehlermeldung aus...warum?

y_train, X_train = get_labels_and_texts("data/train.ft.txt.bz2")
y_test, X_test = get_labels_and_texts("data/test.ft.txt.bz2")
# print(type(train_text))
vectorizer.fit(X_train)
# print(vectorizer.get_feature_names_out()) # test purpose
train_vec = vectorizer.transform(X_train)
test_vec = vectorizer.transform(X_test)


# Step 4: Regularization
model = Sequential()
model.add(layers.Input(shape=(2,)))

model.add(layers.Dense(10, activation="softmax"))
model.add(layers.Dense(20, activation="softmax"))
model.add(layers.Dense(1, activation="softmax"))
model.summary()

model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.fit(train_vec, y_train, epochs=20, batch_size=25)

def evaluate(m):
    print("[INFO] evaluating network...")
    y_pred = m.predict(train_vec)
    print(y_test)
    print(y_pred)
    print("precision: "+ str(precision_score(y_test, y_pred)))
    print("recall: "+ str(recall_score(y_test, y_pred)))
    print("f1: "+ str(f1_score(y_test, y_pred)))

evaluate(model)