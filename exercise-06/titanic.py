import math
import pandas as pd
from tensorflow import keras as ke
from keras.layers import Dense, Input
from keras.activations import softmax as sf
from sklearn.model_selection import train_test_split as split

df = pd.read_csv("data/train.csv")
print("DataFrame loaded.")

df = df.drop(labels=["Name", "PassengerId"], axis=1)
print("DataFrame filtered")

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
print("Dataframe mapped.")


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
print("Incomplete rows eliminated.")

df = df.reset_index().drop("index", 1)

x = df.drop("Survived", 1)
y = df["Survived"]

x_train, x_test, y_train, y_test = split(x, y, train_size=0.8)
print("Dataset prepared.")

model = ke.Sequential()
model.add(Dense(10, activation=sf, name="hidden"))
print("New layer initialized.")
model.add(Input(len(x_train.columns)))
model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])
print("Model compiled.")
model.fit(x_train, y_train)
print("Model trained.")

y_predict = model.predict(x_test)
# The reason why this doesn't work is because the predict method keeps returning
# a very strange list of lists that are almost all the same, which doesn't make the
# slightest bit of sense.

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