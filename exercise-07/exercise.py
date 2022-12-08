import bz2
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer as cv
import keras
from keras.layers import Input, Dense, Dropout
from keras.activations import softmax as sf
from keras.losses import mean_squared_error as msqe
from keras.regularizers import L2


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


def create_model(dropout=False):
    model = keras.Sequential()
    model.add(Input(5000))
    model.add(Dense(5, activation=sf, name="hidden1"))
    if dropout:
        model.add(Dense(3, activation=sf, name="hidden2"))
        model.add(Dropout(0.25))
    else:
        model.add(Dense(8, activation=sf, name="hidden2",
                        activity_regularizer=L2(0.01), bias_regularizer=L2(0.01), kernel_regularizer=L2(0.01)))
    model.compile(loss=msqe, optimizer="sgd", metrics=["accuracy"])
    return model


train_labels, train_texts = get_labels_and_texts("data/train.ft.txt.bz2")
print("Training data retrieved.")
test_labels, test_texts = get_labels_and_texts("data/test.ft.txt.bz2")
print("Test data retrieved.")

vectorizer = cv(max_features=5000, lowercase=True)
print("Vectorizer initialized.")
vectorizer.fit(train_texts)
print("Vectorizer trained.")
train_vec = vectorizer.transform(train_texts)
print("Training vector saved.")
test_vec = vectorizer.transform(test_texts)
print("Test vector saved.")

reg_model = create_model(False)
dropout_model = create_model(True)
print("Regularized model:")
reg_model.evaluate(x=test_vec, y=test_labels)
print("\nDropout model:")
dropout_model.evaluate(x=test_vec, y=test_labels)