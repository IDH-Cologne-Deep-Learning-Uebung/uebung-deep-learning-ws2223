import random

l1 = [[int(random.normalvariate(50,20)) for x in range(random.randint(5,20))] for x in range(10)]

maxValue = 0

for outerlist in l1:
  count = 0
  for element in outerlist:
    count = count + 1
  if (count>maxValue):
    maxValue = count


print(maxValue)
i = 0
for outerlist in l1:
  print(i, end=" ")
  ele = 0
  for element in range(maxValue):
    try:
      print(outerlist[ele], end=" ")
      ele = ele + 1
    except IndexError:
      print("0", end=" ")
  print()
  i = i + 1