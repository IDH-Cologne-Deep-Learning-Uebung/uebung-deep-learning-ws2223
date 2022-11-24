import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")
df.drop(labels=['PassengerId', 'Name'], axis=1, inplace=True)

df['Sex'] = df['Sex'].replace(['male', 'female'], '1', '0')
df['Embarked'] = df['Embarked'].replace(['Q', 'C', 'S'], '0', '1', '2')
df['Cabin'] = df['Cabin'].replace(['B', 'C', 'D', 'E', 'F',' G'], '0', '1', '2', '3', '4', '5')

df["Sex"] = pd.to_numeric(df["Sex"])
df["Embarked"] = pd.to_numeric(df["Embarked"])
df["Cabin"] = pd.to_numeric(df["Cabin"])

x_data = df[['PcClass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
y_data = df['Survived']

df_train = df.sample(frac=0.8, random_state=1)
df_test = df.drop(df_train.index)

regressor = LinearRegression()
regressor.fit(x_data, y_data)

y_pred = regressor.predict(y_data)

recall_score(y_data, y_pred, *['PcClass', 'Sex', 'Age','SibSp' ,'Parch' ,'Ticket' ,'Ticket' ,'Fare' ,'Cabin' ,'Embarked'])
precision_score(y_data, y_pred, *['PcClass', 'Sex','Age','SibSp','Parch','Ticket','Ticket','Fare','Cabin','Embarked'])
f1_score(y_data, y_pred, *['PcClass','Sex','Age','SibSp','Parch','Ticket','Ticket','Fare','Cabin','Embarked'])




# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
# 3. Remove all rows that contain missing values


# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.

