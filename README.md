# Exercise 4: IO, Exceptions, pypi

This is the fourth exercise, and it is concerned with input/output, exceptions and python package management.
As always, the deadline is the beginning of the next session.
## Step 1
If not already done clone this repository to your local computer. On the command line, you would use the following command: `git clone https://github.com/IDH-Cologne-Deep-Learning-Uebung/uebung-deep-learning-ws2223`.
If already done, pull the latest version from our repository and merge it to your branch. This can be achieved by using 
- `git checkout master`
- `git pull`
- `git checkout EIGENER_BRANCH`
- `git merge master`.

## Step 2: IO
Write a new python script called `io.py`. The script reads in the file `wiki.txt`, which contains 10000 randomly selected sentences from the German Wikipedia. The script then filters the sentences, and produces three output files: 

1. The file `short.txt` contains all sentences that contain less than 30 characters. You can use the function [`len()`](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range) to determine the length of a string. 
2. The file `articles.txt` contains all sentences that start with an article (for simplicity, let's just use "Der", "Die", and "Das"). You can use the function [`startswith(...)`](https://docs.python.org/3/library/stdtypes.html#str.startswith) on a string, to find out if it has a certain prefix.
3. The file `april.txt` contains all sentences that contain the string "April". To check wether a string contains another string, you can use the `in`-expression, e.g., `if "April" in sentence:`. 

In all cases, try to use list comprehension.

## Step 3: Exceptions

The file `4.py` contains a full python script and can be executed directly. It produces output such as the following:

```
0 25 52 48 61 32 69 28 51 52 50 46 106 26 11 36 58 
1 52 54 16 4 25 49 56 99 20 36 25 37 23 62 72 59 39 41 53 53 
2 34 49 42 24 67 33 23 64 53 39 39 76 87 31 
3 36 18 62 6 39 41 39 34 87 22 26 50 37 78 34 43 49 24 36 38 
4 59 46 63 54 69 44 88 92 44 51 62 16 40 42 
5 44 67 68 4 83 57 53 69 70 49 
6 41 39 52 41 49 49 27 43 45 41 
7 37 49 56 52 40 14 
8 46 81 55 44 27 54 
9 39 38 54 30 37 50 42 33 28 32 35 70 35 39 14 
```

The first column of the table is a row number. The remaining columns contain between 5 and 20 normally distributed random numbers. Your goal is to implement a simple *padding* mechanism. Padding means that we add zeros to each row, such that each row has the same length in the end. Thus, the output of row 0 should be, for instance, `0 25 52 48 61 32 69 28 51 52 50 46 106 26 11 36 58 0 0 0 0`.

Your padding implementation should use exceptions. The idea is to make a for loop until 20, and try to access the list element at the position. If the list is shorter, we catch the exception and print a zero.

## Step 4: PyPi

For this exercise, you shouldn't need to commit code, but you need to verify that your installation is working as expected. Please install the libraries `numpy`, `scikit-learn`and `pandas`.

You can use the code in `5.py` to test your installation. If everything works, you should see something like this:

```
[[0. 0. 0.]
 [0. 0. 0.]]
Survived    int64
dtype: object
  (0, 276)	1
  (0, 446)	1
  (0, 1156)	1
  (0, 2114)	1
  (0, 2851)	1
  (0, 3167)	1
  (0, 4113)	1
  (0, 4555)	1
  (0, 4905)	1
```

Please pay attention for error messages or exceptions!

## Step 5: Commit
Commit your changes to your local repository and push them to the server.