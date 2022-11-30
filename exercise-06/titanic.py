import pandas as pd
from keras import Sequential, Input, Model
from keras.layers import Dense

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

y = df["Survived"]
x = df.drop("Survived", 1)

inputs = Input(shape=(len(df),))
x = Dense(10, activation='softmax')(inputs)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)
# model.summary()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.1)

model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
model.fit(x_train, y_train)