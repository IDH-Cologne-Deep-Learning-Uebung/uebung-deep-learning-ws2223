# This program takes console input through piping.
# Use cat input.txt | python io.py
import sys

current_sentence = ""
string_input = ""
out_short = ""
out_articles = ""
out_april = ""

for line in sys.stdin:
    sentence_end_position = line.find('.\n')
    # print(sentence_end_position)
    if (sentence_end_position != -1):
        current_sentence += string_input
        string_input = "" 
        current_sentence = line[:sentence_end_position+3]
        if (len(current_sentence) < 30):
            out_short += current_sentence
        if (current_sentence.startswith(('Der', 'Die', 'Das'))):
            out_articles += current_sentence
        if ('April' in current_sentence):
            out_april += current_sentence
        current_sentence = ""
    else: 
        string_input += line[:sentence_end_position]

# print(out_short)
# print(out_articles)
# print(out_april)
with open('short.txt', 'w') as f:
    f.write(out_short)
with open('articles.txt', 'w') as f:
    f.write(out_articles)
with open('april.txt', 'w') as f:
    f.write(out_april)
