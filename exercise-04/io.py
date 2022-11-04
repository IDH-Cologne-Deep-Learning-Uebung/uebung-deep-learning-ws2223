data = open("C:\Users\jan\Documents\uebung-deep-learning-ws2223\exercise-04\wiki.txt")

for line in data.readlines ():
    print(line)
data.close()