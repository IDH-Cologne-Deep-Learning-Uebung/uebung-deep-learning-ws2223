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

print(func1(1,2))
print(func1("Welt", "Hallo"))
print(func1(None, None))
print(func1(11, "Freunde"))
print(func1(5.4, 3.6))

def func2(*args):
    if len(args) < 2:
        return args
    if len(args) == 2:
        return func1(*args)
    if len(args) > 2:
        return len(args)

print(func2("Hello"))
print(func2(1,3))
print(func2(1,2,3,4,5,6))

def func3(**kwargs):
    if 'a' in kwargs and 'b' in kwargs:
        return func1(a, b) # scheitere hier daran, was ich func1 übergeben soll. trage ich kwargs ein, wird das als type an func1 übergeben
    else:
        return func2(*kwargs)
    
print (func3(a=1, b=10, c=7, d=5))