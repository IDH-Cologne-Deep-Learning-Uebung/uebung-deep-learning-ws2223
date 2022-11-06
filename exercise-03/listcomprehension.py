x = [0,1,2,3,4,5,6,7,8,9]
y = [True, False, False, True, "False", False, True, True, True, "True"]

a = [zahl*2 for zahl in x]
b = [zahl for zahl in x if zahl % 2 == 0]
c = [word for word in y if type(word) == bool]
d = [word for word in y if type(word) == str]
e = [[True for zahl in x[:x.index(zahl)]] for zahl in x if zahl % 2 != 0]
