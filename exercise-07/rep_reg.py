import bz2
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import recall_score, precision_score, f1_score
from tensorflow import keras
from tensorflow.keras import Sequential, layers, regularizers

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

v = CountVectorizer(max_features=2, lowercase=True)
y_train, x_train = get_labels_and_texts("data/train.ft.txt.bz2")
y_test, x_test = get_labels_and_texts("data/test.ft.txt.bz2")
v.fit(x_train)

v_train = v.transform(x_train)
v_test = v.transform(x_test)

model1 = keras.Sequential()
model1.add(layers.Input(shape=(2,)))
model1.add(layers.Dense(10, activation="sigmoid", activity_regularizer=regularizers.L2(0.2)))
model1.add(layers.Dense(20))

model1.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model1.fit(v_train, y_train, epochs=100, batch_size=25)

model2 = keras.Sequential()
model2.add(layers.Input(shape=(2,)))
model2.add(layers.Dense(10))
model2.add(layers.Dense(20))
model2.add(layers.Dropout(0.5))

model2.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model2.fit(v_train, y_train, epochs=100, batch_size=25)

def evaluate(model):
    print("[INFO] evaluating network...")
    y_pred = model.predict(v_train, batch_size=25)
    print(y_test)
    print(y_pred)
    print("precision: "+ str(precision_score(y_test, y_pred)))
    print("recall: "+ str(recall_score(y_test, y_pred)))
    print("f1: "+ str(f1_score(y_test, y_pred)))

evaluate(model1)
evaluate(model2)
