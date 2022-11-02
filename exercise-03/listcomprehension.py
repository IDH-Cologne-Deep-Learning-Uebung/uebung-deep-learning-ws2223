x = [0,1,2,3,4,5,6,7,8,9]
y = [True, False, False, True, "False", False, True, True, True, "True"]

a = [i*2 for i in x]
print(a)

b = [i for i in x if i % 2 == 0]
print(b)

c = [i for i in y if i == True]
print(c)

d = [i for i in y if isinstance(i,str)]
print(d)

e = [[True]*i for i in x]
print(e)