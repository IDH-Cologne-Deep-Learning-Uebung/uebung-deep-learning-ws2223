'''
Created on 9 Nov 2022

@author: laryrauh
'''
with open('wiki.txt', 'r') as ifp:
            with open('short.txt', 'w') as ofp:
                for line in ifp:
                    if len(line) < 30:
                        ofp.write(line)


with open('wiki.txt', 'r') as ifp:
            with open('articles.txt', 'w') as ofp:
                for line in ifp: 
                    if line.startswith(('Der', 'Die', 'Das')):
                        ofp.write(line)
                        
                        
with open('wiki.txt', 'r') as ifp:
            with open('april.txt', 'w') as ofp:
                for line in ifp: 
                    if 'April' in line:
                        ofp.write(line)
                        