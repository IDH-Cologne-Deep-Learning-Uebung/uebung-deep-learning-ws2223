data = open("wiki.txt",mode ="r",encoding="UTF-8")

A = []
[A.append(line) for line in data if len(line)<30]

data.close()