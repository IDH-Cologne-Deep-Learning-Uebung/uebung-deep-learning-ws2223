def func1(a, b):
  if type(a) == int and type(b) == int:
    return a+b
  if type(a) == str and type(b) == str:
    return b+" "+a
  if a is None and b is None:
    return "Does not exist."
  if type(a) != type(b):
    return None
  return type(a)

<<<<<<< HEAD
print(func1(1,2))
print(func1("Welt", "Hallo"))
print(func1(None, None))
print(func1(11, "Freunde"))
print(func1(5.4, 3.6))

def func2(*args):
  arglist = []
  for a in args:
    arglist.append(a)
  if len(arglist) < 2:
    return str(len(arglist))
  if len(arglist) == 2:
    return func1(*args)
  if len(arglist) > 2:
    return len(arglist)

print(func2(1))
print(func2(1,2))
print(func2(1,2,3))

def func3(**kwargs):
  if 'a' in kwargs and 'b' in kwargs:
    return func1(kwargs.get('a'), kwargs.get('b'))
  else:
    return func2(kwargs)

print(func3(a=1, b=2))
=======
func1(1,2)
func1("Welt", "Hallo")
func1(None, None)
func1(11, "Freunde")
func1(5.4, 3.6)
>>>>>>> master
