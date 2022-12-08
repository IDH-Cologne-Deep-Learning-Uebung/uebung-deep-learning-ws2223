import bz2
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.metrics import precision_score, recall_score, f1_score

print("TENSORFLOW: " + tf.__version__)
print("PANDAS: " + pd.__version__)

train_file = "data\\train.ft.txt.bz2"
test_file = "data\\test.ft.txt.bz2"

def get_labels_and_texts(file, n=50):
    labels = []
    texts = []
    i = 0
    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        labels.append(int(x[9]) - 1)
        texts.append(x[10:].strip())
        i = i + 1
        if i >= n:
          return texts
    return texts

#(np.array(labels)

vectorizer = CountVectorizer()
train_data = get_labels_and_texts(train_file, n=1500)
#print(train_data)
test_data = get_labels_and_texts(test_file, n=1500)
vectorizer.fit(train_data)
texts_vec = vectorizer.transform(train_data)
#print(texts_vec)

model_1 = keras.Sequential()
model_1.add(layers.Input(shape=(9)))