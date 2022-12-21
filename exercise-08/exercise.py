import bz2

import numpy as np
import json

from tensorflow.python.keras import models, layers
from keras.preprocessing.text import Tokenizer, text_to_word_sequence as tf
from keras.utils import pad_sequences 
from keras import regularizers as re


def get_labels_and_texts(file, n=10000):
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


train_labels, train_texts = get_labels_and_texts('data/train.ft.txt.bz2')

#Let´s set up our Tokenizer
class DataTokenizer(Tokenizer):
  def tokenize (self, inputs):
    return tf.strings.reduce_join(inputs, seperator = "", axis= -1)


Tokenizer = DataTokenizer()
#alright, let´s tokenize
Tokenizer.tokenize('data/train.ft.txt.bz2')

#the tokenizer returns a json object, so let´s convert that into an array we can use
#first, we need to convert it from a json library to a python library:
JsonTokens = json.loads(Tokenizer)

#Then we need to turn that library into an array
FinalTokens = JsonTokens.items()
elemen = list(FinalTokens)
con_arr = np.array(elemen)
#and now we have an array

#one last thing: need to pad the sequence length so they´re all equally long:
sequence = FinalTokens
tf.keras.utils.pad_sequences(
  sequence, maxlen = None, dtype='int32', padding = 'pre', truncating = 'pre', value = 0.0
)

ffnn = models.Sequential()
ffnn.add(layers.Input(shape=('maxlen')))
ffnn.add(layers.Embedding(1000, 200, input_length= 'maxlen'))
ffnn.add(layers.flatten())
ffnn.add(layers.Dense(100, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(50, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(1, activation="sigmoid"))

ffnn.summary()

ffnn.compile(loss="binary_crossentropy", optimizer="sgd",
  metrics=["accuracy"])

ffnn.fit(train_texts, train_labels, epochs=10, batch_size=10, verbose=1)


