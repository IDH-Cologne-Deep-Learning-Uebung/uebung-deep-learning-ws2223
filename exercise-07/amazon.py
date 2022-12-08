import bz2

import numpy as np
from keras import Sequential, layers, regularizers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score


#2
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

print(get_labels_and_texts('data/test.ft.txt.bz2', n=100))


#3
vectorizer = CountVectorizer(max_features=2, lowercase=True)

y_train, X_train = get_labels_and_texts("data/train.ft.txt.bz2")
y_test, X_test = get_labels_and_texts("data/test.ft.txt.bz2")

vectorizer.fit(X_train)
train_vec = vectorizer.transform(X_train)
test_vec = vectorizer.transform(X_test)
print(train_vec)


#4
model = Sequential()
model.add(layers.Input(shape=(2,)))

model.add(layers.Dense(10, activation="softmax", activity_regularizer=regularizers.L2(0.2)))
model.add(layers.Dense(1, activation="softmax"))
model.add(layers.Dense(12, activation="relu"))
model.add(layers.Dropout(0.1))
model.summary()

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(train_vec, y_train, epochs=100, batch_size=25)


def evaluate(model):
    print("[INFO] evaluating network...")
    y_pred = model.predict(train_vec, batch_size=25)
    print(y_test)
    print(y_pred)
    print("precision: " + str(precision_score(y_test, y_pred)))
    print("recall: " + str(recall_score(y_test, y_pred)))
    print("f1: " + str(f1_score(y_test, y_pred)))


evaluate(model)