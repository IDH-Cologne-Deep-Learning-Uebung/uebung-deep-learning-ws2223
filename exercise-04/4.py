import random

l1 = [[int(random.normalvariate(50, 20)) for x in range(random.randint(5, 20))] for x in range(10)]

i = 0
for row in l1:
    print(i, end=" ")
    for element in row:
        print(element, end=" ")
    print()
    i = i + 1

print("\n with Padding: \n")
rows = 0

for row in l1:
    print(rows, end="")
    for index in range(20):
        try:
            row[index]
        except IndexError:
            row.append(0)
    for element in row:
        print(element, end="")
    print()
    rows = rows + 1
