lines = open("wiki.txt").readlines()

short = [line for line in lines if len(line) < 30]
open("short.txt", "w").writelines(short)

articles = [line for line in lines if line.startswith(("Der ", "Die ", "Das ", ))]
open("articles.txt", "w").writelines(articles)

april = [line for line in lines if "April" in line]
open("april.txt", "w").writelines(april)
