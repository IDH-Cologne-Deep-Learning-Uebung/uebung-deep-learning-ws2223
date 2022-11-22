import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")
df.drop (['name', 'PassengerId'], axis = 1) #This deletes the unwanted coloumns

#Let´s convert non numerics into numerics
#Step 1: making sure there´s actual numbers in the columns:
df['Sex'] = df['Sex'].replace(['male', 'female'], '1', '0')
df['Embarked'] = df['Embarked'].replace(['Q', 'C','S'], '0','1','2')
df['Cabin'] = df['Cabin'].replace(['B', 'C','D','E','F','G'], '0','1','2','3','4','5')

df.dropna #Deletes rows with null values in them

#And now let´s convert those Strings of numbers into numeric data types: 
df["Sex"] = pd.to_numeric(df["Sex"])
df["Embarked"] = pd.to_numeric(df["Embarked"])
df["Cabin"] = pd.to_numeric(df["Cabin"])

#Alright, Data´s All prepared, let´s move on to step 4
#First, let´s define the "target" label, in our case wether or not the person survived:
x_data = df[['PcClass','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']]
y_data = df['Survived']

#Now let´s split up the training and test sets
df_train = df.sample(frac=0.8, random_state=1)
df_test = df.drop(df_train.index)

#fitting the data to the linear regressor....
regressor = LinearRegression()
regressor.fit(x_data, y_data)

y_pred = regressor.predict(y_data) #the actual regressor command

#and finally calculating precision/recall and f-score:
recall_score(y_data, y_pred, *['PcClass','Sex','Age','SibSp','Parch','Ticket','Ticket','Fare','Cabin','Embarked'])
precision_score(y_data, y_pred, *['PcClass','Sex','Age','SibSp','Parch','Ticket','Ticket','Fare','Cabin','Embarked'])
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

