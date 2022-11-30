import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")


# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevsant for our problem).
del df["Name"]
del df["PassengerId"]

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
enc = LabelEncoder()
enc.fit(df["Sex"])
df["Sex"] = enc.transform(df["Sex"])
enc.fit(df["Cabin"])
df["Cabin"] = enc.transform(df["Cabin"])
enc.fit(df["Ticket"])
df["Ticket"] = enc.transform(df["Ticket"])
enc.fit(df["Embarked"])
df["Embarked"] = enc.transform(df["Embarked"])

# 3. Remove all rows that contain missing values
df = df.dropna()


# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
labels = df.iloc[:, 0]
input = df.iloc[:, 1:]

# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
input_train, input_test, labels_train, labels_test = train_test_split(input, labels)

# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
logReg = LogisticRegression(solver="liblinear")
logReg.fit(input_train, labels_train)

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
labels_predicted = logReg.predict(input_test)
precision = precision_score(labels_test, labels_predicted) * 100
recall = recall_score(labels_test, labels_predicted) * 100
f1 = f1_score(labels_test, labels_predicted) * 100

print("Data: ")
print(df)
print("Predictions: ")
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-Score: ", f1)


#----------------------------------
# Exercise-06
#----------------------------------
import keras as ks
from keras.layers import Dense, Input

# 1. Set up a network with a single hidden layer of size 10. The hidden and the output layer use the `softmax` activation function. Test and evaluate.

model = ks.Sequential()
model.add(Input(shape=(9,)))
model.add(Dense(10, activation="softmax"))
model.add(Dense(1, activation="softmax"))
model.compile(loss = "mean_squared_error", optimizer = "sgd", metrics = ["accuracy"])          
model.fit(input_train, labels_train)

# 2. Create a network with 2 hidden layers of sizes 20 and 10. The first layer uses a `sigmoid` activation, the second one `relu` (output layer should use `softmax` again).

model2 = ks.Sequential()
model2.add(Input(shape=(9,)))
model2.add(Dense(20, activation="sigmoid"))
model2.add(Dense(10, activation="relu"))
model2.add(Dense(1, activation="softmax"))
model2.compile(loss = "mean_squared_error", optimizer = "sgd", metrics = ["accuracy"])          
model2.fit(input_train, labels_train)

# 3. Experiment with different settings. Can you increase the f-score above 56% ? If not, do you have an idea why not?

labels6_predicted = model.predict(input_test)
labels6_predicted2 = model2.predict(input_test)
print(f1_score(labels_test, labels6_predicted))
print(f1_score(labels_test, labels6_predicted2))
