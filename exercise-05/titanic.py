import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
# 3. Remove all rows that contain missing values
df = df.drop(['Name', "PassengerId"], axis=1)
df[['Sex', 'Cabin', 'Ticket', 'Embarked']] = df[['Sex', 'Cabin', 'Ticket', 'Embarked']].astype(int)
df = df.dropna()
print(df)

# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
y = df.iloc[:, 0] #these are the training labels
x = df.iloc[:, 1:9] # these are the input features

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

lr = LogisticRegression(solver = 'liblinear').fit(x_train, y_train)
y_pred = lr.predict(x_test)
sklearn.metrics.precision_score(y_test, y_pred)
sklearn.metrics.recall_score(y_test, y_pred)
sklearn.metrics.f1_score(y_test, y_pred)
