x = [0,1,2,3,4,5,6,7,8,9]
y = [True, False, False, True, "False", False, True, True, True, "True"]

a = [x*2 for x in x]
print(a)

b = [x for x in x if x % 2 == 0]

b = x[::2]
print(b)

c = [y for y in y if y == True]
print(c)

d = [z for z in y if type(z) == str]
print(d)

e = [[True] * x for x in x if x % 2 != 0]
print(e)

