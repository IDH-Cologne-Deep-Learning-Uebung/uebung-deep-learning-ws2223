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

func1(1,2)
func1("Welt", "Hallo")
func1(None, None)
func1(11, "Freunde")
func1(5.4, 3.6)

def func2(*args):
  if len(args) < 2:
    return str(len(args))
  elif len(args) == 2:
    func1(args[0], args[1])
  return len(args)

def func3(**kwargs):
  if "a" in kwargs and "b" in kwargs:
    return func1(kwargs["a"], kwargs["b"])
  return func2(kwargs.values())
