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
