import bz2
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.python import keras
from keras import layers, regularizers, Sequential


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
max_features = 1000,
lowercase = True
)

y_train, X_train = get_labels_and_texts("data/train.ft.txt.bz2")
y_test, X_test = get_labels_and_texts("data/test.ft.txt.bz2")

vectorizer.fit(X_train)

train_vec = vectorizer.transform(X_train)
test_vec = vectorizer.transform(X_test)

#------------

model = keras.Sequential()
model.add(layers.Input(shape=(2,)))

model.add(layers.Dense(10, activation="softmax"))
model.add(layers.Dense(20, activation="softmax"))

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(train_vec, y_train, epochs=75, batch_size=20)
evaluate(model)

modelv1 = keras.Sequential()
modelv1.add(layers.Input(shape=(2,)))

modelv1.add(layers.Dense(10,
activation="sigmoid",
activity_regularizer=regularizers.L2(0.2)))
modelv1.add(layers.Dense(20))
modelv1.add(layers.Dropout(0.5))

modelv1.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
modelv1.fit(train_vec, y_train, epochs=75, batch_size=20)
evaluate(modelv1)

modelv2 = keras.Sequential()
modelv2.add(layers.Input(shape=(2,)))

modelv2.add(layers.Dense(10))
modelv2.add(layers.Dense(20))
modelv2.add(layers.Dropout(0.5))

modelv2.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
modelv2.fit(train_vec, y_train, epochs=75, batch_size=20)
evaluate(modelv2)

def evaluate(mo):
    y_pred = mo.predict(X_test)
    print(y_test)
    print(y_pred)
    print("precision: " + str(precision_score(y_test, y_pred)))
    print("recall: " + str(recall_score(y_test, y_pred)))
    print("f1: " + str(f1_score(y_test, y_pred)))