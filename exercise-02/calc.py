
def func(sum): + str(num1+num2)
def func(pro): + str(num1*num2)


start: print("Enter number 1: ")
num1 = int(input())
print("Enter number 2: ")
num2 = int(input())
print("sum or pro ?")
give = input()
if give == "sum":
     print("sum: " + str(num1+num2))
if give == "pro":
     print("pro: " + str(num1*num2))
answer = input('you want to exit?: [y/n]')

while answer[0].lower() == 'n':
    print("Restart")
    print("Enter number 1: ")
    num1 = int(input())
    print("Enter number 2: ")
    num2 = int(input())
    print("sum or pro ?")
    give = input()
    if give == "sum":
        print("sum: " + str(num1+num2))
    if give == "pro":
        print("pro: " + str(num1*num2))
    answer = input('you want to exit?: [y/n]')

if answer[0].lower == 'y':
    exit()

