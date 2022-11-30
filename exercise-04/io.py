<<<<<<< HEAD
textfile = open("wiki.txt", "r")
shortfile = open("short.txt", "w")
articlefile = open("articles.txt", "w")
aprilfile = open("april.txt", "w")
  
content = textfile.readlines()

shortlist = [line for line in content if len(line) < 30]
shortfile.writelines(shortlist)
shortfile.close()

articlelist = [line for line in content if line.startswith("Der") or line.startswith("Die") or line.startswith("Das")]
articlefile.writelines(articlelist)
articlefile.close()

aprillist = [line for line in content if "April" in line]
aprilfile.writelines(aprillist)
aprilfile.close()
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
