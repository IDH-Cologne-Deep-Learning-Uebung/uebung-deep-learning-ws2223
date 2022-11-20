import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import presicion_score, recall_score, f1_score

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")


# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).

d1 = df.pop("Name")
d1 = df.pop("PassengerId")
print(d1)


# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".

s = LabelEncoder()
s.fit(df["Sex"])
df["Sex"] = s.transform(df["Sex"])

c = LabelEncoder()
c.fit(df["Cabin"])
df["Cabin"] = c.transform(df["Cabin"])

t = LabelEncoder()
t.fit(df["Ticket"])
df["Ticket"] = t.transform(df["Ticket"])

e = LabelEncoder()
e.fit(df["Embarked"])
df["Embarked"] = e.transform(df["Embarked"])



# 3. Remove all rows that contain missing values

dr.dropna(axis = 0, how = "any", inplace = True)



# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.

x = df.iloc[:, 1:]
y = df.iloc[:, 0]


# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.

y_train, y_test, x_train, x_test = train_test_split(y, x, test_size = None, train_size = None, random_state = None, shuffle = True, stratify = None)


# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.

lr = LogisticRegression(random_state = none, solver = "liblinear", multi_class = "ovr", verbose = 0)
lr.fit(y_train, x_train)
y_predict = lr.predict(x_test)


# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.

prec = presicion_score(y_test, y_predict)*100
rec = recall_score(y_test, y_predict)*100
fscore = f1_score(y_test, y_predict)*100

print("Precision: ", prec, "\nRecall: ", rec, "n\F-Score: ", fscore)



