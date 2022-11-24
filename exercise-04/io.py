
with open("wiki.txt", "r") as f:
    wiki_txt = f.readlines()
    wiki_list = [elem.replace("\n", "") for elem in wiki_txt]

with open("short.txt", "w") as sf:
    short_list =[x for x in wiki_list if len(x)<=30]
    sf.writelines(element + "\n" for element in short_list)

with open("articles.txt", "w") as af:
    articles_list = [elem for elem in wiki_list if elem.startswith("Der") == True or elem.startswith("Das") == True or elem.startswith("Die")==True]
    af.writelines(elem + "\n" for elem in articles_list)

with open("april.txt", "w") as apf:
    april_sentences = [elem for elem in wiki_list if "April" in elem]
    apf.writelines(elem + "\n" for elem in april_sentences)
