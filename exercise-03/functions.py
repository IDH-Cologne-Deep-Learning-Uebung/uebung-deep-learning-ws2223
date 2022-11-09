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

def func2( *Variablen ):
  n = len(Variablen)
  if n < 2:
    return str(n)
  elif n == 2:
    return func1(Variablen)
  else:
    return n

def func3( **Variablen_benannt ):
  if 'a' in Variablen_benannt and 'b' in Variablen_benannt:
    return func1(Variablen_benannt.get('a'),Variablen_benannt.get('b'))
  else:
    return func2(Variablen_benannt)

print(func2("Hallo", 2, 5.5))

print(func3(a = 1, b = 5))

print(func3(a = "Hello"))