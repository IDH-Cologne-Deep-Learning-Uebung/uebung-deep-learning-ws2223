import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")


# ## Step 3


# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
ln = df.drop(labels=['Name', 'PassengerId'], axis=1)


# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
le = LabelEncoder()
s = le.fit_transform(df['Sex'])
c = le.fit_transform(df['Cabin'])
t = le.fit_transform(df['Ticket'])
e = le.fit_transform(df['Embarked])

ln["Sex"] = s
ln["Cabin"] = c
ln["Ticket"] = t
ln["Embarked"] = e


# 3. Remove all rows that contain missing values
ln.dropna(axis = 1)



# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.1)



# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.

clf = LogisticRegression(random_state=0, solver="liblinear")
clf.fit(x_train, y_train)

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
y_pred = classifier.predict(x_test)

print("precision: "+ str(precision_score(y_test, y_pred)))
print("recall: "+ str(recall_score(y_test, y_pred)))
print("f1: "+ str(f1_score(y_test, y_pred)))