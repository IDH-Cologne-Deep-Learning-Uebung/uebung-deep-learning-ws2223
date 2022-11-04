wiki = open("wiki.txt", encoding="utf-8").readlines()

open("short.txt", "w", encoding="utf-8").writelines([line for line in wiki if len(line) < 30])
open("article.txt", "w", encoding="utf-8").writelines([line for line in wiki if line[0:4] in ["Der ", "Die ", "Das "]])
open("april.txt", "w", encoding="utf-8").writelines([line for line in wiki if "April" in line])
