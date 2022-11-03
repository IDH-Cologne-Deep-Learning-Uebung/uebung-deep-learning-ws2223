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

def func2(*kwargs):
  if len(kwargs) == 2:
    a, b = kwargs
    return func1(a, b)
  else :
    return len(kwargs)

def func3(*kwargs):
  if kwargs[0] == "a" and kwargs[1] == "b":
    return True

print(func2(3,3))
# print(func3(a = 1, b = 2))

func1(1,2)
func1("Welt", "Hallo")
func1(None, None)
func1(11, "Freunde")
func1(5.4, 3.6)