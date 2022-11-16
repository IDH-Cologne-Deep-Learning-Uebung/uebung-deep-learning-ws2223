import random

l1 = [[int(random.normalvariate(50,20)) for x in range(random.randint(5,20))] for x in range(10)]

i = 0
for outerline in l1:
    z = len(outerline)
    try:
        l = outerline[19]
    except:
        z = len(outerline)
        outerline.append(0)
        while z < 19:
            z= len(outerline)
            outerline.append(0)

for outerline in l1:
    print(i, end=" ")
    for element in outerline:
        print(element, end=" ")
    print()
    i = i +1

