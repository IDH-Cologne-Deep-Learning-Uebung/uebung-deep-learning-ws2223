# jeder Buchstabe aus dem Wort "word" als Liste mit For-Schleife
l = []
for buchstabe in "word":
    l.append(buchstabe)
print(l)

# jeder Buchstabe aus dem Wort "word" als Liste mit List Comprehension
l = [buchstabe for buchstabe in "word"]
print(l)

# Zahlen aus Liste zahlen falls Zahl kleiner 20 mit For-Schleife und if-Bedingung
zahlen = [1, 17, 45, 13, 9, 57]
l = []
for zahl in zahlen:
    if zahl < 20:
        l.append(zahl)
print(l)

# Zahlen aus Liste zahlen falls Zahl kleiner 20 mit List Comprehension
l = [zahl for zahl in zahlen if zahl < 20]
print(l)

# Zahlen aus Liste zahlen *2 falls Zahl kleiner 20
l = [zahl*2 for zahl in zahlen if zahl < 20]
print(l)

# Zahlen von 1 bis 10
l = [zahl+1 for zahl in range(10)]
print(l)

# Zahlen von 1 bis 10 jeweils * 2
l = [(zahl+1)*2 for zahl in range(10)]
print(l)

# Erstellung einer Matrix mit geschachtelten For-Schleifen
matrix = []
for x in range(5):
    matrix.append([])

    for y in range(5):
        matrix[x].append(y)
print(matrix)

# Erstellung einer Matrix mit nested list comprehension
matrix = [[y for y in range(5)] for x in range(5)]
print(matrix)
