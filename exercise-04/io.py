data = open("wiki.txt",mode ="r",encoding="UTF-8")

kurz = open("short.txt",mode ="w")
[kurz.write(line) for line in data if len(line)<30]
kurz.close()


articles=("Der","Die","Das")
artikel = open("articles.txt",mode ="w")
[artikel.write(line) for line in data if line.startswith(articles) and  line.startswith(" ",3,4)]
artikel.close()


april = open("april.txt",mode = "w")
[april.write(line) for line in data if "April" in line]
april.close()


data.close()