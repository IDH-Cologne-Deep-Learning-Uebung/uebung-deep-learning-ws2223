x = [0,1,2,3,4,5,6,7,8,9]
y = [True, False, False, True, "False", False, True, True, True, "True"]

a = [2*num for num in x]
b = [num for num in x if num % 2 == 0]
c = [num for num in x if num]
d = [s for s in y if type(s) == str]
e = [[True]*el for el in x if el % 2 == 1]