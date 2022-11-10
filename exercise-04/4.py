import random

l1 = [[int(random.normalvariate(50,20)) for x in range(random.randint(5,20))] for x in range(10)]

for outerlist in l1:
  for i in range(20):
    try:
      print(outerlist[i])
    except IndexError:
      outerlist.insert(i, 0)

i = 0
for outerlist in l1:
  print(i, end=" ")
  for element in outerlist:
    print(element, end=" ")
  print()
  i = i + 1

