x = [0,1,2,3,4,5,6,7,8,9]
y = [True, False, False, True, "False", False, True, True, True, "True"]

a = [z*2 for z in x]
print(a)

b = [z for z in x if z%2 == 1]
print(b)

c = [z for z in y if z == True]
print(c)

d = [z for z in y]
print(d)

e = [z*[bool(z)] for z in b]
print(e)
