print("Enter number 1: ")
num1 = int(input())
print("Enter number 2: ")
num2 = int(input())
print("Do you want to use addition or multiplication?")
i = str(input())
if "add" in i:
    print("Sum: " + str(num1+num2))
if "multi" in i:
    print("Product: " + str(num1*num2))

