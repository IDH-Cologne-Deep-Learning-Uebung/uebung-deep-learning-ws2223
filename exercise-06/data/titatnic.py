import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow import keras
from keras.layers import Dense, Input

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

model = keras.Sequential()
model.add(layers.Dense(10, activation="softmax",
                       name="a single hidden layer"))

model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])
model.fit(x_train, y_train)

#2
model.add(layers.Input(shape=(20, activation == "sigmoid")))
model.add(layers.Input(shape=(10, activation == "relu")))

# 3
y_predict = [int(i) for i in model.predict(x_test)]
y_test = [int(i) for i in y_test]

tp, tn, fp, fn = 0, 0, 0, 0
for i in range(0, len(y_predict)):

    if y_predict[i] == 1 and y_test[i] == 1:
        tp += 1

    elif y_predict[i] == 0 and y_test[i] == 0:
        tn += 1

    elif y_predict[i] == 1 and y_test[i] == 0:
        fp += 1

    elif y_predict[i] == 0 and y_test[i] == 1:
        fn += 1


