print("Enter number 1: ")
num1 = int(input())
print("Enter number 2: ")
num2 = int(input())
operator = input("Addition or multiplication? Please type in add or multi:")
if (operator == "add"):
    print("Sum: " + str(num1+num2))
elif (operator == "multi"):
    print("Product: " + str(num1*num2))
