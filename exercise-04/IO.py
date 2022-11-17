Text = open("wiki.txt", mode = 'r', encoding = "utf-8")

Kurz = open("short.txt", mode = 'w', encoding = "utf-8")
[Kurz.write(Zeile) for Zeile in Text if len(Zeile) < 30]
Kurz.close

Artikel = open("articles.txt", mode = 'w', encoding = "utf-8")
Anfang = ["Der", "Die", "Das"]
[Artikel.write(Zeile) for Zeile in Text if len(Zeile) < 30]
#[Artikel.write(Zeile) for Zeile in Text if Zeile.startswith(Anfang)]
Artikel.close

April = open("april.txt", mode = 'w', encoding = "utf-8")
[April.write(Zeile) for Zeile in Text if "April" in Zeile]
April.close

Text.close