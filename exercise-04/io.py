<<<<<<< HEAD
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
=======
# Because we will need this code a few times, 
# we define a function for it
def ListToFile(List, Filename):
  # open file for writing
  file = open(Filename, "w")
  
  # iterate over list
  for Line in List:
    # write list into file
    file.write(Line)

  # close file
  file.close()

# open file for reading
file = open("wiki.txt")

# get a list of all sentences
sentences = file.readlines()

# close file
file.close()

# call function for different subsets of sentences
ListToFile([x for x in sentences if len(x) < 30], 
  "short.txt")
ListToFile([x for x in sentences if x.startswith("Der") or x.startswith("Die") or x.startswith("Das")], 
  "articles.txt")
ListToFile([x for x in sentences if "April" in x], 
  "april.txt")
>>>>>>> master
