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

def func2(*args):
  if len(args) < 2:
    return str(len(args))
  if len(args) == 2:
    func1(args[0],args[1])
  if len(args) > 2:
    return int(len(args))

def func3(*args):
  if (args[0] or args[1]) == "a":
    func1(args)

print(func1(1,2))
print(func1("Welt", "Hallo"))
print(func1(None, None))
print(func1(11, "Freunde"))
print(func1(5.4, 3.6))
print(func2(0.5,0.4))
print(func3("a", "b"))

