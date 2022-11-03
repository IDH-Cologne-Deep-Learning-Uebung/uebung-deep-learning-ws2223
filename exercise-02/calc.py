def calc():
    print("Enter number 1: ")
    num1 = int(input())
    print("Enter number 2: ")
    num2 = int(input())
    print("Choose addition (add) or multiplication (multi)")
    operation = str(input())
    if operation in ["add","addition","Add","Addition"]:
        print("Sum: " + str(num1+num2))
    if operation in ["multi","multiplication","Multi","Multiplication"]:
        print("Product: " + str(num1*num2))
    print("Do this again?")
    answer = str(input())
    if answer in ["y", "Y", "yes", "Yes", "YES"]:
        calc()
    else:
        return print("Thank you")
calc()