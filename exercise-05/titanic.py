import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")


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


# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
x = df.iloc[:, 1:]
y = df.iloc[:, 0]


# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
x_train, x_test, y_train, y_test = train_test_split(x, y)
print("x train: ",x_train)
print("y train: ",y_train)
print("x test: ",x_test)
print("y test: ",y_test)

# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
clf = LogisticRegression(solver="liblinear")
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
pre = precision_score(y_test, y_predict)*100
rec = recall_score(y_test, y_predict)*100
fsc = f1_score(y_test, y_predict)*100

print("Precision: ", pre, "\nRecall: ", rec, "\nF-Score: ", fsc)