fh = open("wiki.txt", encoding="UTF-8")
fs = open("short.txt", "w", encoding="UTF-8")
fa = open("articles.txt", "w", encoding="UTF-8")
fap = open("april.txt", "w", encoding="UTF-8")

for line in fh.readlines():
    if line is not None:
        if len(line) <= 30:
            fs.write(line)
        if line.startswith("Der") or line.startswith("Die") or line.startswith("Das"):
            fa.write(line)
        if "April" in line:
            fap.write(line)


