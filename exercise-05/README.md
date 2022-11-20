# Exercise 5: Logistic Regression with `scikit-learn`

This is the fifth exercise, and it covers initial steps for doing a logistic regression task with scikit-learn. Reading documentation for [`pandas`](https://pandas.pydata.org/docs/reference/index.html) and [`scikit-learn`](https://scikit-learn.org/stable/modules/classes.html) is part of the exercise.

## Step 1
If not already done clone this repository to your local computer. On the command line, you would use the following command: `git clone https://github.com/IDH-Cologne-Deep-Learning-Uebung/uebung-deep-learning-ws2223`.
If already done, pull the latest version from our repository and merge it to your branch. This can be achieved by using 
- `git checkout master`
- `git pull`
- `git checkout EIGENER_BRANCH`
- `git merge master`.

## Step 2: Preparations and data set

The easiest and most straightforward way of doing machine learning experiments is to load data and directly push it into the algorithms. However: Interesting data usually doesn't come in the correct form and format. Thus, you can expect to always spending some time on preparing the data -- sometimes even more time than on the actual experiments.

Typical preparation steps include:

- Reading in CSV files
- Removing columns that we cannot handle or columns that we feel are irrelevant
- Convert features into numeric representations
- Split into train and test data
- Deal with missing values
- Add additional features from other resources

### The data set

This dataset contains information about the titanic passengers, including names, gender, passenger class and whether they survived [the sinking of the ship](https://en.wikipedia.org/wiki/Sinking_of_the_Titanic).

We will use the data set to train a feedforward neural network that predicts — given the other information — whether someone survived. It is therefore a binary classification with the two classes "survived" (encoded as 1) and "drowned" (encoded as 0).

The file `titanic.py` contains initial code to read in the file from disk using a function from the library `pandas`. Using pandas functions, please

1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
3. Remove all rows that contain missing values

All of these can be done with existing `pandas`-functions. Feel free to browse the [documentation](https://pandas.pydata.org/docs/). If you know what you are looking for, you should consult [the API reference](https://pandas.pydata.org/docs/reference/index.html#api).

## Step 3: Machine Learning

We now have a nicely prepared data set and turn to the actual machine learning. This will be done using `scikit-learn`. It is good to look ahead a little bit: Logistic regression is implemented in a class called [`sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression). The class follows the "initialize-fit-predict"-design pattern (like most of the implementations we'll be seeing in this class). Thus: After creating an object of the class, you start the actual training using the `fit()` method. Once trained, you may use the `predict()` method to get new predictions. These methods expect feature values and training labels to be in two separate variabes, often called `x` (features) and `y` (training labels). In addition, you usually want to separate train and test data, thus, you end up with four variables to describe your data: `x_train`, `y_train`, `x_test`, `y_test` (obviously, you can name them differently, but I'd suggest to use a consistent scheme). 

Back to the code:

1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.

## Step 4: Commit
Commit your changes to your local repository and push them to the server.