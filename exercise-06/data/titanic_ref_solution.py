import pandas as pd
import numpy as np
import tensorflow as tf
from keras.applications.densenet import layers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow import keras
from keras.layers import Dense, Input

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df = df.drop("Name", 1)
df = df.drop("PassengerId", 1)

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
def make_numeric(df):
  df["Sex"] = pd.factorize(df["Sex"])[0]
  df["Cabin"] = pd.factorize(df["Cabin"])[0]
  df["Ticket"] = pd.factorize(df["Ticket"])[0]
  df["Embarked"] = pd.factorize(df["Embarked"])[0]
  return df
df = make_numeric(df)

# 3. Remove all rows that contain missing values
df = df.dropna()

# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
y = df["Survived"]
x = df.drop("Survived", 1)

# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.1)

# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.


# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
y_pred = classifier.predict(x_test)

from sklearn.metrics import precision_score, recall_score, f1_score

print("precision: "+ str(precision_score(y_test, y_pred)))
print("recall: "+ str(recall_score(y_test, y_pred)))
print("f1: "+ str(f1_score(y_test, y_pred)))



model = keras.Sequential()
model.add(layers.Input(shape=(9,))
model.add(layers.Dense(10, activation="softmax"))
model.add(layers.Dense(1, activation="softmax"))
model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])
model.fit(x_train, y_train))

model = keras.Sequential()
model.add(layers.Dense(shape=20, activation = "sigmoid"))
model.add(layers.Dense(shape=10, activation = "relu"))
model.add(layers.Dense(shape=1, activation = "softmax"))
model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])
model.fit(x_train, y_train)

y_predict = [int(i) for i in model.predict(x_test)]
y_test = [int(i) for i in y_test]

tp, tn, fp, fn = 0, 0, 0, 0
for i in range(0, len(y_predict)):
    if y_predict[i] and y_test[i] == 1:
        tp += 1
    elif y_predict[i] == 1 and y_test[i] == 0:
        fp += 1
    elif y_predict[i] == 0 and y_test[i] == 1:
        fn += 1
    elif y_predict == 0 and y_test[i] == 0:
        tn +=1
