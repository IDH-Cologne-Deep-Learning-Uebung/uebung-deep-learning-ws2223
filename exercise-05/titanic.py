import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

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

