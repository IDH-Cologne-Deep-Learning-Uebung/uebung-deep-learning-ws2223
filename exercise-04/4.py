import random

l1 = [[int(random.normalvariate(50,20)) for x in range(random.randint(5,20))] for x in range(10)]

i = 0
for outerlist in l1:
  print(i, end=" ")
  for element in outerlist:
    print(element, end=" ")
  try:
      len(outerlist == 20)
  except:
    print("0 " * (20 - len(outerlist)))
  i = i + 1
