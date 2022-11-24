import pandas as pd
import numpy as np
import sklearn

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")
df.pop("Name")
df.pop("PassengerId")
df.dropna(how="any")
#df[["Sex","Cabin","Ticket","Embarked"]] = df[["Sex","Cabin","Ticket","Embarked"]].astype(np.uint64, copy=False)
#print(df["Embarked"])

print(df.head())
#for row in df:
#    for cel in row:
#        if cel == None:
#            df.pop(row)
sklearn.set_config()
#DataFrame.to_csv("output.csv")

# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
# 3. Remove all rows that contain missing values


# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.

