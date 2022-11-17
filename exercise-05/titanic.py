import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
mod_dl = df.pop('Name')
mod_dl = df.pop('PassengerId')
print(mod_dl)

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
le = LabelEncoder()
le.fit(df['Sex'])
df['Sex'] = le.transform(df['Sex'])

la = LabelEncoder()
la.fit(df['Cabin'])
df['Cabin'] = la.transform(df['Cabin'])

li = LabelEncoder()
li.fit(df['Ticket'])
df['Ticket'] = li.transform(df['Ticket'])

lo = LabelEncoder()
lo.fit(df['Embarked'])
df['Embarked'] = lo.transform(df['Embarked'])

# 3. Remove all rows that contain missing values
mod_df = df.dropna()
print("Modified Dataframe : ")
print(mod_df)
mod_df.to_csv('Result_data.csv', index=False)


# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.


# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
from sklearn.model_selection import train_test_split
training_data, testing_data = train_test_split(mod_df,test_size = 0.2, shuffle=False)
print(f" training data: {training_data}")
print(f" test data: {testing_data}")

# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
from sklearn import datasets
clf = LogisticRegressionCV(random_state=None, solver='liblinear', multi_class='ovr', verbose=0)
clf.fit(training_data)
y_pred = clf.predict(testing_data)

#from sklearn import linear_model
#from sklearn.datasets import load_iris
#LRG = linear_model.LogisticRegression(training_data, solver = 'liblinear',multi_class='auto')
#print(f"log: {LRG}")

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
train_rec = recall_score(testing_data, y_pred)*100
train_prec = precision_score(testing_data, y_pred)*100
train_fscore = f1_score(testing_data, y_pred)*100
print("Recall: ", train_rec, "\nPrecision: ", train_prec, "\nFScore: ", train_fscore)