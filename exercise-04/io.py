with open("wiki.txt") as f:
    txt = f.readlines()
    sentences = [sentence.replace("\n", "") for sentence in txt]

with open("short.txt", "w") as f:
    [f.write(sentence + "\n") for sentence in sentences if len(sentence) < 30]

with open("articles.txt", "w") as f:
    [f.write(sentence + "\n") for sentence in sentences if sentence.startswith("Der") or sentence.startswith("Die") or sentence.startswith("Das")]

with open("april.txt", "w") as f:
    [f.write(sentence + "\n") for sentence in sentences if "April" in sentence]
            