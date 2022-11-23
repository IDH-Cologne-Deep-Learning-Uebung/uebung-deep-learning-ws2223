try:
    short = open("short.txt", "x")
    artikel = open("artikel.txt", "x")
    april = open("april.txt", "x")

except IOError:
    print("Files already created")

short = open("short.txt", "w")
short.write("")
artikel = open("artikel.txt", "w")
artikel.write("")
april = open("april.txt", "w")
april.write("")

def readfilewiki(filename="wiki.txt"):
    fo = open(filename, encoding="UTF-8")
    lines = [line for line in fo.readlines()]
    fo.close()
    return lines

def analysis(lines):
    for line in lines:
        length = len(line)
        firstThree = line[0:3]
        tokens = line.split()
        if length < 30:
            short = open("short.txt", "a", encoding="UTF-8")
            short.writelines(line)
            short.close()
        if firstThree == "Der" or firstThree == "Die" or firstThree == "Das":
            artikel = open("artikel.txt", "a", encoding="UTF-8")
            artikel.writelines(line)
            artikel.close()
        for token in tokens:
            if token == "April" or token == "april":
                april = open("april.txt", "a", encoding="UTF-8")
                april.writelines(line)
                april.close()

analysis(readfilewiki())
