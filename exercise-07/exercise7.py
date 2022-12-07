import bz2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras import layers
from keras.backend import clear_session


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

labels, texts = get_labels_and_texts('train.ft.txt.bz2')
test_lbls, test_texts = get_labels_and_texts('test.ft.txt.bz2')

vectorizer = CountVectorizer()

vectorizer.fit(texts)

X_train = vectorizer.transform(texts)
X_test = vectorizer.transform(texts)

# classifier = LogisticRegression()
# classifier.fit(X_train, labels)
# score = classifier.score(X_test, test_lbls)
# print("SKlearn-Accuracy:", score)

# Sklearn accuracy lies within 50%

### REGULARIZATION
def reg_model(X_train, X_test, labels, test_lbls):
    from keras.regularizers import L2
    input_dim = X_train.shape[1]

    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='sigmoid', activity_regularizer=L2(0.2)))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()

    history = model.fit(
        X_train, labels,
        epochs=10,
        verbose=False,
        validation_data=(X_test, test_lbls),
        batch_size=10
    )

    clear_session()

    loss, accuracy = model.evaluate(X_train, labels, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, test_lbls, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

### DROPOUT
def dropout_model(X_train, X_test, labels, test_lbls):
    input_dim = X_train.shape[1]

    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(20))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()

    history = model.fit(
        X_train, labels,
        epochs=10,
        verbose=False,
        validation_data=(X_test, test_lbls),
        batch_size=10
    )

    clear_session()

    loss, accuracy = model.evaluate(X_train, labels, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, test_lbls, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

# reg_model(X_train, X_test, labels, test_lbls)
dropout_model(X_train, X_test, labels, test_lbls)