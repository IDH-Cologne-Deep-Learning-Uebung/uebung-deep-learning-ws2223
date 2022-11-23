import pandas as pd

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

df = df.drop(columns=["Name", "PassengerId"])
df = df.dropna()
print(df.columns)
# print(df["Sex"].unique())
df["Sex"] = df["Sex"].map({'male': 0, 'female': 1})
unique_cabin = df["Cabin"].unique()

cabin_numeric = dict(zip(unique_cabin, range(len(unique_cabin))))
df["Cabin"] = df["Cabin"].map(cabin_numeric)

ticket_numeric = dict(zip(df["Ticket"].unique(), range(len(df["Ticket"].unique()))))
print()
# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
# 3. Remove all rows that contain missing values


# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.

