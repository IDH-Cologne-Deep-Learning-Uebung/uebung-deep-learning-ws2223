Ifile = open("wiki.txt", encoding="UTF-8")
shortFile = open("short.txt", mode="w", encoding="UTF-8")
articlesFile = open("articles.txt", mode="w", encoding="UTF-8")
aprilFile = open("april.txt", mode="w", encoding="UTF-8")
for line in Ifile.readlines():
    if len(line) < 30:
        shortFile.write(line)
    elif line.startswith("Der ") or line.startswith("Die ") or line.startswith("Das "):
        articlesFile.write(line)
    elif  "April" in line:
        aprilFile.write(line)