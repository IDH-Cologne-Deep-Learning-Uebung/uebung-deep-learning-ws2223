<<<<<<< HEAD
def readFile():
    wikiFile = open("wiki.txt", mode="r", encoding="utf-8")
    shortFile = open("short.txt", mode="w")
    articlesFile = open("articles.txt", mode="w")
    aprilFile = open("april.txt", mode="w")
    
    shortFile.writelines(sL for sL in wikiFile if len(sL) < 30)
    shortFile.close()
    
    articlesFile.writelines([aL for aL in wikiFile if aL.startswith("Der " or "Die " or "Das ")])
    articlesFile.close()
    
    aprilFile.writelines([apL for apL in wikiFile if "April" in apL])
    aprilFile.close()
    
readFile()
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
<<<<<<< HEAD:exercise-04/io.py
ListToFile([x for x in sentences if len(x) < 30], 
  "short.txt")
ListToFile([x for x in sentences if x.startswith("Der") or x.startswith("Die") or x.startswith("Das")], 
  "articles.txt")
ListToFile([x for x in sentences if "April" in x], 
  "april.txt")
>>>>>>> master
=======
ListToFile([x for x in sentences if len(x) < 30],
           "short_ref_solution.txt")
ListToFile([x for x in sentences if x.startswith("Der") or x.startswith("Die") or x.startswith("Das")],
           "articles_ref_solution.txt")
ListToFile([x for x in sentences if "April" in x],
           "april_ref_solution.txt")
>>>>>>> master:exercise-04/io_ref_solution.py
