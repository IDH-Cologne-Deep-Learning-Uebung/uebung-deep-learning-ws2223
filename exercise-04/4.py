import random

l1 = [[int(random.normalvariate(50,20)) for x in range(random.randint(5,20))] for x in range(10)]

i = 0
for outerlist in l1:
  print(i, end=" ")
  for element in outerlist:
    print(element, end=" ")
  print()
  i = i + 1

try:
  for element in l1:
    print(str.center(50, "0"))
except:
  print("no")