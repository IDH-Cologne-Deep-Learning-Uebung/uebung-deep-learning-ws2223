import pandas as pd

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# ## Step 3 
# Preparations copied from last exercise
df = df.drop("Name", 1)
df = df.drop("PassengerId", 1)

def make_numeric(df):
  df["Sex"] = pd.factorize(df["Sex"])[0]
  df["Cabin"] = pd.factorize(df["Cabin"])[0]
  df["Ticket"] = pd.factorize(df["Ticket"])[0]
  df["Embarked"] = pd.factorize(df["Embarked"])[0]
  return df
df = make_numeric(df)

df = df.dropna()
y = df["Survived"]
x = df.drop("Survived", 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
  random_state=0, test_size=0.1)



# ## Step 4
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from sklearn.metrics import precision_score, recall_score, f1_score
# because we're doing it twice, wrap the evaluation into a function
def evalu(mo):
  y_pred = mo.predict(x_test)
  print(y_test)
  print(y_pred)
  print("precision: "+ str(precision_score(y_test, y_pred)))
  print("recall: "+ str(recall_score(y_test, y_pred)))
  print("f1: "+ str(f1_score(y_test, y_pred)))

# model 1
model_1 = keras.Sequential()
model_1.add(layers.Input(shape=(9)))
model_1.add(layers.Embedding(input_dim=9, output_dim=50))
model_1.add(layers.Dense(10, activation="softmax"))
model_1.add(layers.Dense(1, activation="softmax"))
model_1.summary()

model_1.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
model_1.fit(x_train, y_train, epochs=20, batch_size=25)

evalu(model_1)

# model 2
# model_2 = keras.Sequential()
# model_2.add(layers.Input(shape=(9)))
# model_2.add(layers.Dense(20, activation="sigmoid"))
# model_2.add(layers.Dense(10, activation="relu"))
# model_2.add(layers.Dense(1, activation="softmax"))
# model_2.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
# model_2.fit(x_train, y_train, epochs=20, batch_size=25)
#
# evalu(model_2)

