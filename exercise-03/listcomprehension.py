x = [0,1,2,3,4,5,6,7,8,9]
y = [True, False, False, True, "False", False, True, True, True, "True"]

a = [i * 2 for i in x]
b = [i for i in x if i%2 == 0]

c = [n for n in y if n]
d = [n for n in y if isinstance(n, str)]

e = [i*[bool(i)] for i in x if not i%2==0]

print(f"a = {a}\nb = {b}\nc = {c}\nd = {d}\ne = {e}\n")
