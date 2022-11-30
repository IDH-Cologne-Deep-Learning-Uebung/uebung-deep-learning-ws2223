import pandas as pd

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

df = df.drop(columns=["Name", "PassengerId"])
# print(df["Sex"].unique())
df["Sex"] = df["Sex"].map({'male': 0, 'female': 1})
unique_cabin = df["Cabin"].unique()

cabin_numeric = dict(zip(unique_cabin, range(len(unique_cabin))))
df["Cabin"] = df["Cabin"].map(cabin_numeric)

ticket_numeric = dict(zip(df["Ticket"].unique(), range(len(df["Ticket"].unique()))))
df["Ticket"] = df["Ticket"].map(ticket_numeric)

# One expression
df["Embarked"] = df["Embarked"].map(
    dict(zip(
        df["Embarked"].unique(),
        range(len(df["Embarked"].unique()))
    ))
)
df = df.dropna()

# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
# 3. Remove all rows that contain missing values
from sklearn.model_selection import train_test_split
from keras import Sequential, layers
from keras.activations import softmax
Y_col = 'Survived'
X_cols = df.loc[:, df.columns != Y_col].columns
X_Train, X_test, y_train, y_test = train_test_split(
    df[X_cols], df[Y_col], 
    test_size=0.20, random_state=66
    )

inputs = Input((len(X_cols),))
hidden = layers.Dense(10, activation=softmax)
outputs = layers.Dense(activation=softmanx)(hidden)
model = Sequential(inputs, outputs)
print(model.summary())
