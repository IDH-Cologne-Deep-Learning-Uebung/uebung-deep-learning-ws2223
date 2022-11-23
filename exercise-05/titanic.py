import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df.drop(["Name", "PassengerId"], axis = 1, inplace = True)
# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
label = preprocessing.LabelEncoder()
df["Sex"] = label.fit_transform(df["Sex"])
df["Embarked"] = label.fit_transform(df["Embarked"])
df["Cabin"] = label.fit_transform(df["Cabin"])
df["Ticket"] = label.fit_transform(df["Ticket"])
# 3. Remove all rows that contain missing values
df.dropna(inplace = True)

print(df)

# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
x = df.iloc[:, 1:]
y = df.iloc[:, 0]
# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
lr = LogisticRegression(solver="liblinear", random_state=0).fit(x_train,y_train)
predict = lr.predict(x_test)
# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
precision = precision_score(y_test, predict)*100
recall = recall_score(y_test, predict)*100
f = f1_score(y_test, predict)*100
