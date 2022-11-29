import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras import layers
from keras.layers import activation

# read the data from a CSV file (included in the repository)
df = pd.read_csv("train.csv")

# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df = df.drop(columns=["Name", "PassengerId"])

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
sex = LabelEncoder()
sex.fit(df["Sex"])
df["Sex"] = sex.transform(df["Sex"])

cab = LabelEncoder()
cab.fit(df["Cabin"])
df["Cabin"] = cab.transform(df["Cabin"])

tick = LabelEncoder()
tick.fit(df["Ticket"])
df["Ticket"] = tick.transform(df["Ticket"])

emb = LabelEncoder()
emb.fit(df["Embarked"])
df["Embarked"] = emb.transform(df["Embarked"])

# 3. Remove all rows that contain missing values
df.dropna(axis=0, how="any", inplace=True)
print(df)

# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
x = df.iloc[:, 1:]
y = df.iloc[:, 0]

# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
x_train, x_test, y_train, y_test = train_test_split(x, y)

# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
clf = LogisticRegression(solver="liblinear")
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
pre = precision_score(y_test, y_predict) * 100
rec = recall_score(y_test, y_predict) * 100
fsc = f1_score(y_test, y_predict) * 100

print("Precision: ", pre, "\nRecall: ", rec, "\nF-Score: ", fsc)

# Exercise 6
# 1. Set up a network with a single hidden layer of size 10. The hidden and the output layer use the softmax activation function. Test and evaluate.
model = keras.Sequential()
model.add(layers.Dense(10, activation="softmax"))

model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])
model.fit(x_train, y_train)

# 2. Create a network with 2 hidden layers of sizes 20 and 10. The first layer uses a sigmoid activation, the second one relu (output layer should use softmax again).
model.add(layers.Input(shape=(20, activation == "sigmoid")))
model.add(layers.Input(shape=(10, activation == "relu")))

# 3. Experiment with different settings. Can you increase the f-score above 56% ? If not, do you have an idea why not?
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