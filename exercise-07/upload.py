# step 2
import bz2

import keras as keras
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras import regularizers


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


# step 3
#vectorizer = CountVectorizer(max_features=10, lowercase=True)

#y_train, X_train = get_labels_and_texts("data/train.ft.txt.bz2")
#y_test, X_test = get_labels_and_texts("data/test.ft.txt.bz2")
#vectorizer.fit(X_train)

#train_vector = vectorizer.transform(X_train)
#test_vector = vectorizer.transform(X_test)

# step 4
#model1 = Sequential()
#model1.add(layers.Input(shape=(2,)))

#model1.add(layers.Dense(1, activation="softmax"))
#model1.add(layers.Dropout(0.5))
#model1.summary()

#model1.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
#model1.fit(train_vector, y_train, epochs=100,  batch_size=256)


#def evalua(model1):
 #   y_pred = model1.predict(train_vector,  batch_size=256)
 #   print(y_test)
 #   print(y_pred)
 #   print("precision: " + str(precision_score(y_test, y_pred)))
#    print("recall: " + str(recall_score(y_test, y_pred)))
 #   print("f1: " + str(f1_score(y_test, y_pred)))


#evalua(model1)
