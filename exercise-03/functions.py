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

def func2(*arguments):
    if len(arguments)<2:
        return str(len(arguments))
    if len(arguments) == 2:
        return func1(*arguments)
    if len(arguments)>2:
        return int(len(arguments))
        

print(func2(5))
print(func2(4,5,6,7))

#**kwargs == arbitary number of keyword arguments
def func3(**kwargs):
    if kwargs == "b" and kwargs == "a":
        return func1(**kwargs)
    #if kwargs = !b and kwargs == !a:
    else:
        return func2(**kwargs) 
        
        
#testing
print(func2(5))
print(func2(5,3))
print(func2(4,5,6,7))