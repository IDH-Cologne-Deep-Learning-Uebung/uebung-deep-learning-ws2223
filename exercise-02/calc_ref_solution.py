# if false, we will exit the main loop
cont = True

# as long as cont is True, we loop
while cont:
  print("Enter number 1: ")
  num1 = int(input())
  print("Enter number 2: ")
  num2 = int(input())
  print("Enter operation:")
  op = input()
  # check wether op is contained in a list of strings
  if op in ["addition", "add", "+"]:
    print("Sum: " + str(num1+num2))
  if op in ["multiplication", "multi", "*"]:
    print("Product: " + str(num1*num2))
  
  # reuse variable name with a different type
  cont = input("Continue (Y/N)? ")
  cont = cont in ["Y", "J", "Yes", "Ja", "y", "j"]
