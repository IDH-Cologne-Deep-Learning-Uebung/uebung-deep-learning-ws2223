
import numpy as np
import csv
from tensorflow import keras
from tensorflow.keras import layers

csvfile = open("train.csv", "r")

train = csvfile
train = train.reshape ([4,25])
y_train = train[0]
x_train = np.rot90(train[1:])

model = keras.Sequential()
model.add(layers.Input(shape=(3,)))
model.add(layers.Dense(5, activation="sigmoid"))
model.add(layers.Dense(1, activation="softmax"))

model.compile(loss="mean_squared_error",optimizer="sgd",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=100,batch_size=5)