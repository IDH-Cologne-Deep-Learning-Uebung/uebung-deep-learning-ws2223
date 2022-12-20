import bz2

import numpy as np
from keras.layers import Embedding
from keras.utils import pad_sequences
from tensorflow import keras
from tensorflow.python.keras import models, layers
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras import regularizers


def get_labels_and_texts(file, n=1000):
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
y_test, test_texts = get_labels_and_texts('data/train.ft.txt.bz2')

# Preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1
train_sequences = tokenizer.texts_to_sequences(train_texts)

MAX_LENGTH = max(len(train_ex) for train_ex in train_sequences)

train_texts = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding="post")

test_sequences = tokenizer.texts_to_sequences(test_texts)
x_test = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding="post")

# pre-trained Embeddings:
embeddings_dict = {}  # This dictionary will contain all the words available in the glove embedding file.
with open("data/glove.6B/glove.6B.100d.txt", 'r', encoding="utf-8") as f:
  for line in f:
      values = line.split()  # split lines at white space
      word = values[0]  # word equals first element
      vector = np.asarray(values[1:], "float32")  # rest of line convert to numpy arr = vector of word position
      embeddings_dict[word] = vector  # update dict with word + vector

print('##### GloVe data loaded ######')

EMBEDDING_DIM = embeddings_dict.get('a').shape[0]

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    embedding_vect = embeddings_dict.get(word)
    if embedding_vect is not None:
        embedding_matrix[i] = embedding_vect

embedding_layer = Embedding(vocab_size, EMBEDDING_DIM, embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=False)

print(embedding_matrix.shape)


ffnn = models.Sequential()
ffnn.add(layers.Input(shape=MAX_LENGTH))
ffnn.add(layers.Embedding(vocab_size, 200, input_length=MAX_LENGTH))
ffnn.add(layers.Flatten())
ffnn.add(layers.Dense(100, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(50, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(1, activation="sigmoid"))

ffnn.summary()

# Shuffle the data
seed = 1337
rng = np.random.RandomState(seed)
rng.shuffle(train_texts)
rng = np.random.RandomState(seed)
rng.shuffle(train_labels)

# Extract a training & validation split
validation_split = 0.2
num_validation_samples = int(validation_split * len(train_texts))
x_train = train_texts[:-num_validation_samples]
x_val = train_texts[-num_validation_samples:]
y_train = train_labels[:-num_validation_samples]
y_val = train_labels[-num_validation_samples:]


ffnn.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

ffnn.fit(train_texts, train_labels, epochs=10, batch_size=10, verbose=1)

results = ffnn.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)
