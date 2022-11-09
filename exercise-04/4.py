import random

l1 = [[int(random.normalvariate(50,20)) for x in range(random.randint(5,20))] for x in range(10)]

i = 0
for outerlist in l1:
  print(i, end=" ")
  for element in outerlist:
    print(element, end=" ")
  x = 0
  while x < 20:
    try:
      outerlist[x]
    except:
      print("00", end=" ")
    finally:
      x = x+1
  print()
  i = i + 1