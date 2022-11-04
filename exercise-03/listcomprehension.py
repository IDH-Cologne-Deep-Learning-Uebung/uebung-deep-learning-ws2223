x = [0,1,2,3,4,5,6,7,8,9]
y = [True, False, False, True, "False", False, True, True, True, "True"]

a = [i*2 for i in x]
b = [i for i in x if i % 2 == 0]
c = [i for i in y if i]
d = [i for i in y if isinstance(i,str)]
e = [[True for i2 in range(i)] for i in x]

print(a)
print(b)
print(c)
print(d)
print(e)