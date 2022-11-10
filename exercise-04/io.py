with open("wiki.txt", encoding="utf-8", mode="r") as f:
    lines = f.readlines()
    sentences = []
    for sentence in lines:
        sentences.append(sentence.replace("\n",""))
    print(sentences)
#   print(lines)
with open("short.txt", encoding="utf-8", mode="w") as shorts:
    for line in sentences:
        if len(line) < 30:
            shorts.write(line + "\n")#

articles = ("Der", "Die", "Das")

with open("articles.txt", encoding="utf-8", mode="w") as arts:
    for line in sentences:
        if line.startswith(articles[2]):
            arts.write(line + "\n")

with open("april.txt", encoding="utf-8", mode="w") as april:
    for line in sentences:
        if "April" in line:
            april.write(line + "\n")

#f.close()