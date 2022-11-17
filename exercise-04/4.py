import random

l1 = [[int(random.normalvariate(50,20)) for x in range(random.randint(5,20))] for x in range(10)]

i = 0
for outerlist in l1:
  print(i, end=" ")
  
  
  for z in range(20):
      try:
          outerlist[z]
      except:
          outerlist.append(0)
          
          
  for element in outerlist:
    print(element, end=" ")
  print()
  i = i + 1
  
 
print(len(outerlist))