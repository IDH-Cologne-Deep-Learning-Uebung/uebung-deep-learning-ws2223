import bz2

import numpy as np

from tensorflow.python.keras import models, layers
from keras.layers import Embedding
from tensorflow import keras
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import pad_sequences
from keras import regularizers


def get_labels_and_texts(file, n=1000): # change n
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


train_labels, train_texts = get_labels_and_texts('../exercise-07/data/train.ft.txt.bz2')

# Preprocessing
tokenizer = Tokenizer() # initialize the Tokenizer object
tokenizer.fit_on_texts(train_texts) #each word is assigned a unique number & every word is now represented by a number
vocab_size = len(tokenizer.word_index) + 1 # create vocab index
sequences = tokenizer.texts_to_sequences(train_texts) # convert each sentence into a sequence of numbers 
MAX_LENGTH = max(len(train_ex) for train_ex in sequences)
train_texts = pad_sequences(sequences, maxlen = MAX_LENGTH, padding = "post") # data

# word_index = tokenizer.word_index
# print('###### Found %s unique tokens. ######' % len(word_index))

embeddings_dict = {} # This dictionary will contain all the words available in the glove embedding file.
with open("data/glove.6B.100d.txt", 'r', encoding="utf-8") as f:
  for line in f:
      values = line.split() #split lines at white space
      word = values[0] # word equals first element (0th)
      vector = np.asarray(values[1:], "float32") # rest of line convert to numpy arr = vector of word position
      embeddings_dict[word] = vector # update dict with word + vector

print('###### GloVe data loaded #######')

EMBEDDING_DIM = embeddings_dict.get('a').shape[0]

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word,i in tokenizer.word_index.items():
    embedding_vect = embeddings_dict.get(word)
    if embedding_vect is not None:
        embedding_matrix[i] = embedding_vect



embedding_layer = Embedding(
    vocab_size,
    EMBEDDING_DIM,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)

print(embedding_matrix.shape)

ffnn = models.Sequential()
ffnn.add(layers.Input(shape=(MAX_LENGTH))) 
# ffnn.add(layers.Embedding(vocab_size, EMBEDDING_DIM, weights = [embedding_matrix], input_length=MAX_LENGTH))
ffnn.add(embedding_layer)
ffnn.add(layers.Flatten())
ffnn.add(layers.Dense(100, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(50, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(1, activation="sigmoid"))

ffnn.summary()

ffnn.compile(loss="binary_crossentropy", optimizer="sgd",
  metrics=["accuracy"])

ffnn.fit(train_texts, train_labels, epochs=10, batch_size=10, verbose=1)