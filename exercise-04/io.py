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
