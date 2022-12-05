import bz2

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import recall_score, precision_score, f1_score
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers
from keras.models import Sequential



# Step 2
def get_labels_and_texts(file, n=12000):
    labels = []
    texts = []
    i = 0
    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        lables.append(int(x[9]) - 1)
        texts.append(x[10:].strip())
        i = i + 1
        if i >= n:
            return np.array(labels), texts
        return np.array(lables), texts
        
# Step 3: Representing Text
vectorizer = CountVectorizer(max_features=10, lowercase=True)
x_train, y_train = get_labels_and_texts("data/train.ft.txt.bz2")
x_test, y_test = get_labels_and_texts("data/test.ft.txt.bz2")
vectorizer.fit(x_train)

train_vector = vectorizer.transform(x_train)
test_vector = vectorizer.transform(x_test)

# Step 4: Regularization
model = Sequential()
model.add(layer.Input(shape=(2,)))
model.add(layers.Dense(2, activation="softmax"))
model.add(layers.Dense(14, activation="relu"))
model.add(layers.Dropout(0.5))
model.summary()

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(train_vector, y_train, epochs=100, batch_size=256)

def evaluate(model):
    y_pred = model.predict(train_vector, batch_size=256)
    print(y_test)
    print(y_pred)
    print("recall: " + str(recall_score(y_test, y_pred)))
    print("precision: " + str(precision_score(y_test, y_pred)))
    print("f1: " + str(f1_score(y_test, y_pred)))
    
    evaluate(model)
    
