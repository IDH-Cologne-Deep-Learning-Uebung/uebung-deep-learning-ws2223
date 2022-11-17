open("wiki.txt") as f:
wikiText = f.readlines()
rep = [rep.replace("\n" + "") for rep in wikiText]    

open("short.txt", "w") as f:
[f.write(rep + "\n") for rep in rep (if len(rep) < 30)]

open("articles.text", "w") as f:
[f.write(rep + '\n') for rep in rep if rep.startswith('Der' or 'Das' or 'Die') ]
 
open("april.txt", "w") as f:
[f.write(rep + '\n') for rep in rep if 'April' in rep]