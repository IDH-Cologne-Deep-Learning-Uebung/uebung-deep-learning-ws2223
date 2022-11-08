w = open("wiki.txt", mode="r", encoding="UTF-8").readlines()
open("short.txt", mode="w", encoding="UTF-8").writelines([line for line in w if len(line) < 30])
open("april.txt", mode="w", encoding="UTF-8").writelines(line for line in w if "April" in line)
open("articles.txt", mode="w", encoding="UTF-8").writelines([line for line in w if line[0:4] in ["Der ", "Die ", "Das "]])
