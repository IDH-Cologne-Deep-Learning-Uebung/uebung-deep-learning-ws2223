from lib2to3.fixer_util import String
x = [0,1,2,3,4,5,6,7,8,9]
y = [True, False, False, True, "False", False, True, True, True, "True"]

a = [zahl*2 for zahl in x] 
print(a)

b = [z for z in x if z % 2 == 0]
print(b)

c = [e for e in y if e == True]
print(c)

d = [e for e in y if type(e) == str]
print(d)

e = [[True for _ in range(z)] for z in x if z % 2 != 0]
print(e)

