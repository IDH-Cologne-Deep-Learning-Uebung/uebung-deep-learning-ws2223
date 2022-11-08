data = open("wiki.txt",mode ="r",encoding="UTF-8")

A = []
for line in data.readlines ():
    if len(line)<30:   
        A.append(line)
    else:
        None
data.close()