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
  a = [a1 for a1 in args]
  print(a)
  numArgs = len(args)
  intReturner = 0
  if numArgs < 2:
    return "There are " + str(numArgs) + " arguments"
  if numArgs == 2:
    return func1(args[0], args[1])
  if numArgs > 2:
    for x in args:
      intReturner += x
    return intReturner

  def func3(**kwargs):
    if (kwargs == "a" and kwargs == "b"):
      func1(**kwargs)
    else:
      func2(**kwargs)
