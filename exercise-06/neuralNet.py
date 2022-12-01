import pandas as pd
import numpy as np
from tensorflow import keras

from keras import layers
from keras import metrics
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

#y_pred = regressor.predict(y_data) #the actual regressor command
#I commented out my regressor command because we don´t need it for this exercise, but I wanted to keep it in 
#just so I know where it was

#instead: let´s do this with Keras: 
model = keras.Sequential()
model.add (input(shape = 3,))
model . add ( layers . Dense (5 , activation =" sigmoid "))
model . add ( layers . Dense (1 , activation =" softmax "))

#compiling model:
model . compile ( loss =" mean_squared_error ", optimizer =" sgd ", metrics =[" accuracy "])

#and finally, let´s train it:
model . fit ( x_data , y_data , epochs =100 , batch_size =5)

#last step, metrics: 



#and finally calculating precision/recall and f-score:
#recall_score(y_data, y_pred, *['PcClass','Sex','Age','SibSp','Parch','Ticket','Ticket','Fare','Cabin','Embarked'])
#precision_score(y_data, y_pred, *['PcClass','Sex','Age','SibSp','Parch','Ticket','Ticket','Fare','Cabin','Embarked'])
#f1_score(y_data, y_pred, *['PcClass','Sex','Age','SibSp','Parch','Ticket','Ticket','Fare','Cabin','Embarked'])



