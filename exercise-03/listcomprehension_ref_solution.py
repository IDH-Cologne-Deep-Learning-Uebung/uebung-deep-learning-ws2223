x = [0,1,2,3,4,5,6,7,8,9]
y = [True, False, False, True, "False", False, True, True, True, "True"]

# A list called `a` that contains all elements of `x`, multiplied by 2.
a = [element * 2 for element in x]

# A list called `b` that contains all even elements of `x`.
b = [element for element in x if element % 2 == 0]

# A list called `c` that contains all truish elements of `y` (i.e., all elements that evaluate to `True` in a boolean context)
c = [element for element in y if element]

# A list called `d` that contains all string elements of `y`
d = [element for element in y if type(element) == str]

# A list called `e` that contains a list for each odd element of `x`. The number of list elements is given by the current number of `x`, and all inner list elements are `True`. Use I.e., the list `e` starts like this: `[[True], [True, True, True], [True, True, True, True, True], ...]`
e = [[True]*element for element in x if element % 2 == 1]

