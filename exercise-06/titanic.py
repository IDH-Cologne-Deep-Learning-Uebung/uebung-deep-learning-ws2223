import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df = df.drop(labels="Name", axis=1)
df = df.drop(labels="PassengerId", axis=1)

# Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
def make_numeric(df):
  df["Sex"] = pd.factorize(df["Sex"])[0]
  df["Cabin"] = pd.factorize(df["Cabin"])[0]
  df["Ticket"] = pd.factorize(df["Ticket"])[0]
  df["Embarked"] = pd.factorize(df["Embarked"])[0]
  return df
df = make_numeric(df)

# Remove all rows that contain missing values
df = df.dropna()

# split the input features from the training labels. This can be done easily with `pandas`.
y = df["Survived"] # labels
X = df.drop(labels="Survived", axis=1) # features

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

###########################################################################################
model = Sequential()
model.add(layers.Input(shape=(9,)))

# 1
# Set up a network with a single hidden layer of size 10. 
# The hidden and the output layer use the softmax activation function. 
# Test and evaluate.
# model.add(layers.Dense(10, activation="softmax")) # hidden layer
# model.add(layers.Dense(1, activation="softmax")) # output

# model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=100, batch_size=32)
# # evaluate the network
# print("[INFO] evaluating network...")
# model.evaluate(x_train, y_train, batch_size=5)
# y_pred1 = model.predict(x_test, batch_size=5)

# print("precision: "+ str(precision_score(y_test, y_pred1)))
# print("recall: "+ str(recall_score(y_test, y_pred1)))
# print("f1: "+ str(f1_score(y_test, y_pred1)))


# 2
# Create a network with 2 hidden layers of sizes 20 and 10. 
# The first layer uses a sigmoid activation, 
model.add(layers.Dense(20, activation="sigmoid"))
# the second one relu 
model.add(layers.Dense(10, activation="relu"))
# output layer should use softmax again
model.add(layers.Dense(20, activation="relu"))
# output layer should use softmax again
model.add(layers.Dense(1, activation="softmax")) # output layer

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=100, batch_size=256)
# evaluate the network
print("[INFO] evaluating network...")
y_pred = model.predict(x_test, batch_size=256)

print("precision: "+ str(precision_score(y_test, y_pred)))
print("recall: "+ str(recall_score(y_test, y_pred)))
print("f1: "+ str(f1_score(y_test, y_pred)))

# Feststellung: fscore ändert sich nie ?!