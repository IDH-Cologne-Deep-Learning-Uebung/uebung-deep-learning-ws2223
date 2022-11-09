x = [0,1,2,3,4,5,6,7,8,9]
y = [True, False, False, True, "False", False, True, True, True, "True"]

a = [n*2 for n in x]
print(a)

b = [gerade for gerade in x if gerade % 2 == 0]
print(b)

c = [wahr for wahr in y if wahr == True]
print(c)

d = [text for text in y if isinstance(text,str) == True]
print(d)

e = [[True] * ungerade for ungerade in x if ungerade % 2 == 1]
print(e)