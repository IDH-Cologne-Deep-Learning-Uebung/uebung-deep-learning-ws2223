# Exercise 8: Embeddings

This is the eighth exercise, and it is about input representation using embeddings. The deadline for this task is 22.12.2022.

## Step 1
If not already done clone this repository to your local computer. On the command line, you would use the following command: `git clone https://github.com/IDH-Cologne-Deep-Learning-Uebung/uebung-deep-learning-ws2223`.
If already done, pull the latest version from our repository and merge it to your branch. This can be achieved by using 
- `git checkout master`
- `git pull`
- `git checkout EIGENER_BRANCH`
- `git merge master`.

We will re-use the same data set from last weeks exercise. You can copy or move it into the data directory.

## Step 3: Preprocessing
The file `exercise.py` contains a template for this week. You first need to add code for the preprocessing: Use the keras-`Tokenizer` to tokenize the `train_texts`, convert it to an integer array and pad it to a fixed length. 

## Step 4: Embedding

The network as defined below should be ready to train. However, it uses the integer sequences from before directly as input features, which means we're missing out on generalization (think: overfitting). Add an embedding layer to train embeddings. Don't forget to flatten after the embedding.

Results should be around 50% accuracy, which is disappointing for a binary classification task. The key to improve results here is a larger training set or leveraging pre-trained embeddings. Try out one of these options. If you opt for a larger training set, you might want to leave your computer running all night or even over the weekend. You can use the `n` parameter of the `get_labels_and_texts()`-function to control the size of the data set loaded.

## Step 6: Commit
Commit your changes to your local repository and push them to the server.