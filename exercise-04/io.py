<<<<<<< HEAD

with open("wiki.txt") as f:
    txt = f.readlines()
    sentences = [sentence.replace("\n", "") for sentence in txt]

with open("short.txt", "w") as f:
    [f.write(sentence + "\n") for sentence in sentences if len(sentence) < 30]

with open("articles.txt", "w") as f:
    [f.write(sentence + "\n") for sentence in sentences if sentence.startswith("Der") or sentence.startswith("Die") or sentence.startswith("Das")]

with open("april.txt", "w") as f:
    [f.write(sentence + "\n") for sentence in sentences if "April" in sentence]
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
