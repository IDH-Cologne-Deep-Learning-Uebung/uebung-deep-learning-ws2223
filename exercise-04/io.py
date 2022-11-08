with open("wiki.txt") as f:
    txt = f.readlines()
    line = [line.replace("\n", "") for line in txt]

with open("short.txt", "w") as f:
    [f.write(line + "\n") for line in line if len(line) < 30]

with open("articles.txt", "w") as f:
    [f.write(line + "\n") for line in line if line.startswith("Der") or line.startswith("Die") or line.startswith("Das")]

with open("april.txt", "w") as f:
    [f.write(line + "\n") for line in line if "April" in line]