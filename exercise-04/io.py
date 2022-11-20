<<<<<<< HEAD
data = open("wiki.txt",mode ="r",encoding="utf-8")
#with open("wiki.txt",mode ="r",encoding="ANSI") as data:
kurz = open("short.txt",mode ="w",encoding = "utf-8")
#    with open("short.txt",mode ="w",encoding="ANSI") as kurz:
[kurz.write(line) for line in data if len(line)<30]

articles=("Der","Die","Das")
artikel = open("articles.txt",mode ="w",encoding="utf-8")

#    with open("articles.txt",mode ="w",encoding="ANSI") as artikel:
[artikel.write(line) for line in data if line.startswith(articles) and  line.startswith(" ",3,4)]


april = open("april.txt",mode = "w",encoding="utf-8")
#    with open("april.txt",mode = "w",encoding="ANSI") as april:
[april.write(line) for line in data if "April" in line]

data.close
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
