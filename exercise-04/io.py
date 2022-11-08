#open
file=open("exercise-04\wiki.txt",mode="r",encoding="ANSI", newline="\n")
wiki=[line for line in file]
file.close()
#Sort
short=[line.removesuffix("\n") for line in wiki if len(line)<30]
article=[line.removesuffix("\n") for line in wiki if line.startswith("Der") or line.startswith("Die") or line.startswith("Das")]
april=[line.removesuffix("\n") for line in wiki if line.__contains__("April")]
#Write
shorttxt=open("exercise-04/short.txt",mode="w",errors="ignore")
shorttxt.writelines(short)
articletxt=open("exercise-04/article.txt",mode="w",errors="ignore")
articletxt.writelines(article)
apriltxt=open("exercise-04/aril.txt",mode="w",errors="ignore")
apriltxt.writelines(april)
shorttxt.close()
articletxt.close()
apriltxt.close()