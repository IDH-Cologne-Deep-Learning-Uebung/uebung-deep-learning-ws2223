import random

l1 = [[int(random.normalvariate(50, 20)) for x in range(random.randint(5, 20))] for y in range(10)]

i = 0
for outerlist in l1:
    print(i, end=" ")
    outerlist += [0 for i in range(20-len(outerlist))]
    for element in outerlist:
        print(element, end=(" " * (3-len(str(element)))))
    print()
    i = i + 1
