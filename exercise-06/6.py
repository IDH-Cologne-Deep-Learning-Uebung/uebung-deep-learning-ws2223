import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow import keras
from keras.layers import Dense, Input, activation

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
# del df['Name', 'PassengerId']
df = df.drop(columns=['Name', 'PassengerId'])
print(df)

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
for col in ["Sex", "Cabin", "Ticket", "Embarked"]:
    df[col] = pd.factorize(df[col])[0]
# 3. Remove all rows that contain missing values
df = df.dropna()
print(df)

# ## Step 4
# 1. As a next step, we need to split the input features x from the training labels y. This can be done easily with `pandas`.
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)

# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
clf = LogisticRegression(random_state=None, solver='liblinear', multi_class='ovr', verbose=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
train_prec = precision_score(y_test, y_pred)*100
train_rec = recall_score(y_test, y_pred)*100
train_fscore = f1_score(y_test, y_pred)*100
print("Recall: ", train_rec, "\nPrecision: ", train_prec, "\nfScore: ", train_fscore)

# exercise 06
# 1.
model = keras.Sequential()
model.add(layers.Dense(10, activation="softmax", name="hiddenLayer"))
model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])
model.fit(X_train, y_train)

#2.
model.add(layers.Input(shape=(20, activation=="sigmoid")))
model.add(layers.Input(shape=(10, activation=="relu")))

#3.
y_predict = [int(i) for i in model.predict(X_test)]
y_test = [int(i) for i in y_test]

tp, tn, fp, fn = 0, 0, 0, 0
for i in range(0, len(y_predict)):
    if y_predict[i] == 1 and y_test[i] == 1:
        tp += 1
    elif y_predict[i] == 1 and y_test[i] == 0:
        fp += 1
    elif y_predict[i] == 0 and y_test[i] == 1:
        fn += 1
    elif y_predict[i] == 0 and y_test[i] == 0:
        tn += 1

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)