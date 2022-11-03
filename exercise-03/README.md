# Exercise 3: List Comprehension, Functions

This is the third exercise, and it verifies that you're able to apply list comprehension, implement a function and read in from/write out to files.

## Step 1
If not already done clone this repository to your local computer. On the command line, you would use the following command: `git clone https://github.com/IDH-Cologne-Deep-Learning-Uebung/uebung-deep-learning-ws2223`.
If already done, pull the latest version from our repository and merge it to your branch. This can be achieved by using 
- `git checkout master`
- `git pull`
- `git checkout EIGENER_BRANCH`
- `git merge master`.


## Step 2: List Comprehension
In the file `listcomprehension.py`, you find two pre-defined lists `x` and `y`.
Use list comprehension to define the following new lists:

- A list called `a` that contains all elements of `x`, multiplied by 2.
- A list called `b` that contains all even elements of `x`.
- A list called `c` that contains all truish elements of `y` (i.e., all elements that evaluate to `True` in a boolean context)
- A list called `d` that contains all string elements of `y`
- A list called `e` that contains a list for each odd element of `x`. The number of list elements is given by the current number of `x`, and all inner list elements are `True`. Use I.e., the list `e` starts like this: `[[], [True, True], [True, True, True, True], ...]`


## Step 3: Functions
The file `functions.py` contains a function `func1` that takes two arguments and returns different values depending on these arguments. The function calls below the definition showcase each return value. Read and understand why this works.

Write a function `func2` that takes an arbitrary number of arguments. In the function, check the number of arguments. If there are less than two arguments, return a string with the number of arguments. If there are two arguments, pass them to `func1`. If there are more than two arguments, return the number of arguments as an int value. 

Next, write a function `func3` that takes an arbitrary number of named arguments. Check that two of these arguments have the names `a` and `b`. If so, pass them into `func1`. If not, pass all names of the arguments into `func2`.

If you haven't figured it out by now: These functions don't make sense, they are just exercise material.


## Step 4: Commit
Commit your changes to your local repository and push them to the server.