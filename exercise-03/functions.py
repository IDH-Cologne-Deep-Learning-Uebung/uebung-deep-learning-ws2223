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
<<<<<<< HEAD
func1(5.4, 3.6)

def func2(*args):
  NumberOfArguments = len(args)
    if NumberOfArguments < 2:
        return "The Number of Arguments is " + NumberOfArguments
    if NumberOfArguments == 2:
        return func1(args[0], args[1])
    if NumberOfArguments > 2:
        return NumberOfArguments
        
def func3(**kwargs):
    if (kwargs == "a" and kwargs == "b"):
      return func1(**kwargs)
    else:
      return func2(**kwargs)
=======
func1(5.4, 3.6)
>>>>>>> master
