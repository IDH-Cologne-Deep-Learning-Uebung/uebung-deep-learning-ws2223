x = [0,1,2,3,4,5,6,7,8,9]
y = [True, False, False, True, "False", False, True, True, True, "True"]

a = [x*2 for x in x]
print(a)
b = [x for x in x]
print(b)
c = [y for y in y if y == True]
print(c)
d = [y for y in y if isinstance(y, str)]
print(d)
e = [[True for e in range (x-1)] for x in x if x%2 ==1]
print(e)
