import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df.drop(['Name', 'PassengerId'], axis=1, inplace=True) 
    # Axis is initialized either 0 or 1. 
    # 0 is to specify row and 1 is used to specify column. 
    # Here we have set axis as 1 so that we can delete the required column, 
    # if we wanted to delete a row then axis should be set to 0.

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
le = LabelEncoder()
def convert_to_num(column_name):
    label = le.fit_transform(df[column_name])
    df.drop(column_name, axis=1, inplace=True)
    df[column_name] = label

convert_to_num('Sex')
convert_to_num('Cabin')
convert_to_num('Ticket')
convert_to_num('Embarked')
# Source: https://www.geeksforgeeks.org/how-to-convert-categorical-string-data-into-numeric-in-python/

# 3. Remove all rows that contain missing values
df.dropna(axis=0, how='any', inplace=True ) #If any NA values are present, drop that row 
# print(df.head())

# ## Step 4
# 1. As a next step, we need to split the input features (x) from the training labels (y). This can be done easily with `pandas`.
X = df.iloc[:, 1:] # selects each entries from features (so everything except Survived) 
y = df.iloc[:, 0] # selects entries from row "Survived"
# Source: https://stackoverflow.com/questions/57174196/how-to-split-feature-and-label

# print("Features: ")
# print(x.head())
# print("Labels: ")
# print(y.head())

# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=None, train_size=None, random_state=42, shuffle=True, stratify=None)

# 3. initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
clf = LogisticRegression(random_state=None, solver='liblinear', multi_class='ovr', verbose=0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test) # Predict class labels for samples in X.

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
train_rec = recall_score(y_test, y_pred)*100 
train_prec = precision_score(y_test, y_pred)*100
train_fscore = f1_score(y_test, y_pred)*100
print("Recall: ", train_rec, "\nPrecision: ", train_prec, "\nFScore: ", train_fscore)