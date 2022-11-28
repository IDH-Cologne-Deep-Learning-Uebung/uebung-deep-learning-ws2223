<<<<<<< HEAD
with open('wiki.txt') as f:
     txt = f.readlines()
     s = [s.replace('\n', '') for s in txt]
     

with open('short.txt', 'w') as f:
     [f.write(s + '\n') for s in s
      if len(s) < 30]

with open('articles.text', 'w') as f:
     [f.write(s + '\n') for s in s if s.startswith('Der')
      or s.startswith('Die') or s.startswith('Das')]

with open('april.txt', 'w') as f:
     [f.write(s + '\n') for s in s
      if 'April' in s]
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
