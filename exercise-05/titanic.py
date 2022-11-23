import pandas as pd
import numpy as np
from sklearn import  linear_model
from sklearn.metrics import precision_score,f1_score,recall_score

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df.pop('Name')
df.pop('PassengerId')

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex" x, "Cabin" x, "Ticket" and "Embarked" x.
N = len(df['Sex'])
for i in range(N):

    s = df['Sex'][i]
    if  s == 'male':
        df['Sex'][i] = 0
    elif s == 'female':
        df['Sex'][i] = 1
    else:
        None 

    e = df['Embarked'][i]
    if  e == 'S':
        df['Embarked'][i] = 19
    elif e == 'C':
        df['Embarked'][i] = 3
    elif e == 'Q':
        df['Embarked'][i] = 17     
    else:
        None

    c = df['Cabin'][i] #if cabin is known, 1, else 0

    if type(c) == str:
        df['Cabin'][i] = 1
    else:
        df['Cabin'][i] = 0    


df.pop('Ticket') #Ticket has too many NaN entrys and seems unhelpful for our means

# 3. Remove all rows that contain missing values
df=df.dropna()

# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
target = ['Survived']

# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.

def TrainTestSplit(D,features,split = 0.2):
    T = int(len(D)*split) #Länge des Testanteils

    return [df[features][:T],df[target][:T]],[df[features][T:],df[target][T:]] #Erst Test, dann Trainingsintervall


# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.

def fit_model(x_train,y_train,x_test,y_test,regrtype= linear_model.LogisticRegression(penalty='none')):
    if type(x_train)== pd.core.frame.DataFrame:
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

    
    model = regrtype
    model.fit(x_train,y_train)
    y_pred= model.predict(x_test)
    return model.coef_,y_pred



data_train,data_test= TrainTestSplit(df,features)


coef,y_pred= fit_model(data_train[0],data_train[1],data_test[0],data_test[1])
y_test = data_test[1]

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
print('precision score:', precision_score(y_test,y_pred),'\n', 'f1 score :', f1_score(y_test,y_pred),'\n','recall score:', recall_score(y_test,y_pred))
