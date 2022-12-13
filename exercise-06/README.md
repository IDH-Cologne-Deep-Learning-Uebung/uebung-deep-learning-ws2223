# Exercise 6: Feed-Forward Neural Networks

This is the sixth exercise, and it covers designing and training simple neural networks. We will use the [keras API](https://keras.io): You find [introductory guides](https://keras.io/guides/) as well as the [reference documentation](https://keras.io/api/) on its web page.

## Step 1 
If not already done clone this repository to your local computer. On the command line, you would use the following command: `git clone https://github.com/IDH-Cologne-Deep-Learning-Uebung/uebung-deep-learning-ws2223`.
If already done, pull the latest version from our repository and merge it to your branch. This can be achieved by using 
- `git checkout master`
- `git pull`
- `git checkout EIGENER_BRANCH`
- `git merge master`.


## Step 2: Preparations and data set

We will again use the titanic data set from exercise 5. That means, we can re-use all preparatory work. You can use your own code, or use the one given in the reference solution, which is provided in the file `titanic_ref_solution.py`.


## Step 3: Deep Learning

Again, we can re-use some of the code from last week, in particular the split and test routine.

As the ML core, however, we no longer want to use `scikit-learn` implementation of logistic regression, but a feed-forward neural network. The first question we need to find out: What is the input shape? If you know this, please:

1. Set up a network with a single hidden layer of size 10. The hidden and the output layer use the `softmax` activation function. Test and evaluate.
2. Create a network with 2 hidden layers of sizes 20 and 10. The first layer uses a `sigmoid` activation, the second one `relu` (output layer should use `softmax` again).
3. Experiment with different settings. Can you increase the f-score above 56%? If not, do you have an idea why not?


## Step 4: Commit
Commit your changes to your local repository and push them to the server.