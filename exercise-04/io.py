#data = open("wiki.txt",mode ="r",encoding="UTF-8")
with open("wiki.txt",mode ="r",encoding="utf-8") as data:
    #kurz = open("short.txt",mode ="w")
    with open("short.txt",mode ="w",encoding="utf-8") as kurz:
        [kurz.write(line) for line in data if len(line)<30]


    articles=("Der","Die","Das")
    #artikel = open("articles.txt",mode ="w")

    with open("articles.txt",mode ="w",encoding="utf-8") as artikel:
        [artikel.write(line) for line in data if line.startswith(articles) and  line.startswith(" ",3,4)]


    #april = open("april.txt",mode = "w")
    with open("april.txt",mode = "w",encoding="utf-8") as april:
        [april.write(line) for line in data if "April" in line]

