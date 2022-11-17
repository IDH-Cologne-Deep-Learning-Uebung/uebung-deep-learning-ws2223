import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df.drop(["Name", "PassengerId"], axis=1, inplace=True)

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

le = LabelEncoder()

ticket = le.fit_transform(df["Ticket"])
df.drop("Ticket", axis=1, inplace=True)
df["Ticket"] = ticket

cabin = le.fit_transform(df["Cabin"])
df.drop("Cabin", axis=1, inplace=True)
df["Cabin"] = cabin

embarked = le.fit_transform(df["Embarked"])
df.drop("Embarked", axis=1, inplace=True)
df["Embarked"] = embarked
                  
# 3. Remove all rows that contain missing values
df.dropna(inplace=True)

# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
x = df.iloc[:,1:]
y = df["Survived"]

# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
clf = LogisticRegression(solver="liblinear").fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
print('Precision: ', precision_score(y_pred=y_pred, y_true=y_test))
print('Recall: ', recall_score(y_pred=y_pred, y_true=y_test))
print('F1: ', f1_score(y_pred=y_pred, y_true=y_test))
print('Accuracy: ', accuracy_score(y_pred=y_pred, y_true=y_test))

