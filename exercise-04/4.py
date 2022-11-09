import random

l1 = [[int(random.normalvariate(50,20)) for x in range(random.randint(5,20))] for x in range(10)]

for num in range(20):
  for line in l1:
    try:
      line[num]
    except IndexError:
      line.append(f"{0:02d}")


i = 0
for outerlist in l1:
  print(i, end=" ")
  for element in outerlist:
    print(element, end=" ")
  print()
  i = i + 1




