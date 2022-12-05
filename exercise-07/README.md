# Exercise 7: Text Representation and Combatting Overfitting

This is the seventh exercise, and it has two parts. The first part is about representing textual data in neural networks, and the second one is about overfitting.

## Step 1
If not already done clone this repository to your local computer. On the command line, you would use the following command: `git clone https://github.com/IDH-Cologne-Deep-Learning-Uebung/uebung-deep-learning-ws2223`.
If already done, pull the latest version from our repository and merge it to your branch. This can be achieved by using 
- `git checkout master`
- `git pull`
- `git checkout EIGENER_BRANCH`
- `git merge master`.

## Step 2: Download data
For legal and size reasons, the data is not included in the repository. You need to download the two files `train.ft.txt.bz2` (440MB) and `test.ft.txt.bz2` (50MB) from Ilias instead, and store them in the folder `data` in the repository. The files contain reviews from amazon.com and accompanying sentiment scores (positive/negative). **Beware: Do not uncompress the files.**

The following function can be used to read in the texts. Feel free to include it in your repository (after having understood how it works).

```python
import bz2

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

```

With this function, you can inspect the files, without actually uncompressing them on your disk. The parameter `n` controls how many lines to extract.

## Step 3: Representing Text

Convert the text in the files on a bag of words representation. You can use the [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) of the `scikit-learn` library. Make sure to establish the vocabulary on the training data only, but apply the vectorization to both training and test.


## Step 4: Regularization
Design and train a neural network with two hidden layers. Play around with different layer sizes etc. a little bit, until you have a configuration that works well. Now create (at least) two variants of the network: One with regularization, and one with dropout. Compare the performance score during training and on the test set.

## Step 5: Commit
Commit your changes to your local repository and push them to the server.