with open('wiki.txt') as f:
     txt = f.readlines()
     s = [s.replace('\n', '') for s in txt]
     

with open('short.txt', 'w') as f:
     [f.write(s + '\n') for s in s
      if len(s) < 30]

with open('articles.text', 'w') as f:
     [f.write(s + '\n') for s in s if s.startswith('Der')
      or s.startswith('Die') or s.startswith('Das')]

with open('april.txt', 'w') as f:
     [f.write(s + '\n') for s in s
      if 'April' in s]
