#The script reads in the file wiki.txt, which contains 10000 randomly selected sentences from the German Wikipedia
fo = open("wiki.txt", encoding="utf-8").readlines()
   
fs = open("short.txt", mode= "w", encoding ="utf-8") 
far = open("artikel.txt", mode= "w", encoding="utf-8")
fa = open("april.txt", mode="w", encoding="utf-8")

#The file short.txt contains all sentences that contain less than 30 characters    
fs.writelines(line for line in fo if len(line)< 30)
fs.close()

#The file articles.txt contains all sentences that start with an article, added blank space so we don't get words like Dieser oder Deren
far.writelines(line for line in fo  if line.startswith("Der ") or line.startswith("Die ") or line.startswith("Das "))
far.close()

#The file april.txt contains all sentences that contain the string "April"
fa.writelines(line for line in fo if "April " in line)
fa.close()