import math
from sklearn.model_selection import train_test_split as split
from sklearn.linear_model import LogisticRegression as LogReg

import pandas as pd

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")


# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
# 3. Remove all rows that contain missing values

df = df.drop(labels=["Name", "PassengerId"], axis=1)

sexdict = {
    "male":0,
    "female":1
}

df["Sex"] = df["Sex"].map(sexdict)


def string_to_number(string):
    if string.isdigit():
        return int(string)
    else:
        return_value = ""
        for digit in string:
            if digit.isdigit():
                return_value += digit
        if len(return_value) == 0:
            return 0
        else:
            return int(return_value)


df["Ticket"] = df["Ticket"].map(string_to_number)


def float_to_int(float1):
    if isinstance(float1, str):
        return string_to_number(float1)
    elif math.isnan(float1):
        return float1
    else:
        return int(float1)


df["Cabin"] = df["Cabin"].map(float_to_int)

embarkeddict = {
    "S": 0,
    "C": 1,
    "Q": 2,
    math.nan: 3
}


df["Embarked"] = df["Embarked"].map(embarkeddict)


def has_missing_values(row):
    for value in row:
        if isinstance(value, str):
            pass
        elif math.isnan(value):
            return True
    return False


for row_number in df.index.to_numpy():
    if has_missing_values(df.loc[row_number]):
        df = df.drop(row_number,0)

df = df.reset_index().drop("index", 1)

# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.

x = df.drop("Survived", 1)
y = df["Survived"]

x_train, x_test, y_train, y_test = split(x, y, train_size=0.8)

log_reg = LogReg()
log_reg.fit(x_train, y_train)
y_predict = [int(i) for i in log_reg.predict(x_test)]
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