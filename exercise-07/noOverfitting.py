from sklearn.feature_extraction.text import CountVectorizer
import bz2
import numpy as np
from tensorflow import keras
from keras import layers
from keras.metrics import Accuracy



#This let´s me read the files without having to de-compress them
def get_labels_and_texts(file, n=12000):
    labels = []
    texts = []
    i = 0
    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        labels.append(int(x[9]) - 1)
        texts.append(x[10:].strip())
        i = i + 1
        if i >= n:
          return np.array(labels), texts
    return np.array(labels), texts

#Now, let´s read that thing and define it´s conent as df
df_train = get_labels_and_texts(data/train.ft.txt.bz2, n= 10)

Vectorizer = CountVectorizer(max_features=1000)#This turns the corpus into a bag of words
Lowercase= True #makes everything lowercaste so I don´t have to worry about case sensitivity

#fit the data to the Vectorizer.
Vectorizer.fit(df_train)



#Split it into training and testing data like we did for the titanic
df_train = df.sample(frac=0.8, random_state=1)
df_test = df.drop(df_train.index)

#creating a vocabulary for the training data
texts_vec = Vectorizer.transform(df_train)

#so much for the data, now let´s build us a model

model = keras.Sequential()
model.add (input(shape = 3,))
model . add ( layers . Dense (5 , activation =" sigmoid "))
model . add ( layers . Dense (1 , activation =" softmax "))

#compiling model....
model . compile ( loss =" mean_squared_error ", optimizer =" sgd ", metrics =[" accuracy "])

#let´s train....
model . fit ( df_train , df_test , epochs =100 , batch_size =5)

#aaaand some metrics
m = df.keras.metrics.Accuracy(df_test)
m.update_state([df_test])
m.result().numpy()
#if this is not fixed by the time you´re looking at it I straight up forgot because I now have to
#do exercise 8