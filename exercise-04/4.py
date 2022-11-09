import random

l1 = [[int(random.normalvariate(50,20)) for x in range(random.randint(5,20))] for x in range(10)]

i = 0
for outerlist in l1:
  
  print(i, end=" ")
  j=0
  for element in outerlist:
    print(element, end=" ")
    j+=1
  while j<20:
    print(0,end = " ")
    j=j+1
  print()
  i = i + 1
 