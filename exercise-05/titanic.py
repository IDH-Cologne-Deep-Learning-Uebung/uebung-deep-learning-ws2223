import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")
print(df)

# ##Step 3
# #1.
#df.drop("Name", inplace=True, axis=1)
del df["Name"]
del df["PassengerId"]

# #2. (first replaced single things, so they could be converted to number
#df["Sex"] = df["Sex"].astype(int)
df["Sex"] = df["Sex"].replace({"male": "0", "female": "1"})
df["Ticket"] = df["Ticket"].replace("[^0-9]", "", regex=True)
df["Cabin"] = df["Cabin"].replace({"A": "1.0", "B": "2.0", "C": "3.0", "D": "4.0", "E": "5.0", "F": "6.0", "G": "7.0", "T": "8"}, regex=True)
df["Embarked"] = df["Embarked"].replace({"C": "1", "Q": "2", "S": "3"}, regex=True)
df["Sex"] = pd.to_numeric(df["Sex"], errors="coerce")
df["Cabin"] = pd.to_numeric(df["Cabin"], errors="coerce")
df["Ticket"] = pd.to_numeric(df["Ticket"], errors="coerce")
df["Embarked"] = pd.to_numeric(df["Embarked"], errors="coerce")

# #3.
df = df.dropna()



# ##Step 4
# #1.
y = np.array(df["Survived"])
X = np.array(df.drop(["Survived"], inplace=True, axis=1))

# #2.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# #3.
classifier = LogisticRegression(random_state=0, solver="liblinear")
classifier.fit(X_train, y_train)


# #4.
y_pred = classifier.predict(x_test)
print("precision: "+ str(precision_score(y_test, y_pred)))
print("recall: "+ str(recall_score(y_test, y_pred)))
print("f1: "+ str(f1_score(y_test, y_pred)))
