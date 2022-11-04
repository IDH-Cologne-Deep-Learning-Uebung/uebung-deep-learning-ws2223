x = [0,1,2,3,4,5,6,7,8,9]
y = [True, False, False, True, "False", False, True, True, True, "True"]

a = [2*xi for xi in x]
b = [xi if xi %2 == 1]
c = [yi if yi == True]
d = [yi if type(yi)==str]

ex =[]
for xi in x:
    temp=[]
    for i in len(xi):
        temp.append(True)
    ex.append(temp)
e = [ex[xi] if xi %2 == 0] #Comment