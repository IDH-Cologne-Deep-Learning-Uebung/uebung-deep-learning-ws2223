import random

l1 = [[int(random.normalvariate(50,20)) for x in range(random.randint(5,20))] for x in range(10)]

i = 0
for outerlist in l1:
  print(i, end=" ")
  for x in range(20):
    try:
      print(outerlist[x], end=" ")
    except IndexError:
      print("0", end=" ")
  print()
  i = i + 1
