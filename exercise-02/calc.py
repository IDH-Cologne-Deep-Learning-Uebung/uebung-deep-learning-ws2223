go = True
while go:
 print("Enter number 1: ")
 num1 = int(input())
 print("Enter number 2: ")
 num2 = int(input())
 print("Select function: (a)dd or (m)ultiply")
 sel = str(input())
 if sel == "a" or sel == "add" or sel == "addition":
  print("Sum: " + str(num1+num2))
 elif sel == "m" or sel == "multi" or sel == "multiplication":
  print("Product: " + str(num1*num2))
 print("Go again - (y)es or no? ")
 rep = str(input())
 if rep == "y" or rep =="yes":
    go = True
 else:
    go = False