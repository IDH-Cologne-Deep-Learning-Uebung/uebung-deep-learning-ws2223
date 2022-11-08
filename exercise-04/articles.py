data = open("wiki.txt",mode ="r",encoding="UTF-8")

A = []
articles=("Der","Die","Das")

[A.append(line) for line in data if line.startswith(articles) and  line.startswith(" ",3,4)]

data.close()