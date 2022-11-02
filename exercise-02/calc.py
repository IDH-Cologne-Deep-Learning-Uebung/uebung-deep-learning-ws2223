from sys import exit
import os
import sys

print("Enter number 1: ")
num1 = int(input())
print("Enter number 2: ")
num2 = int(input())

#asking the user which operations he wants to use
choose = input('Do you want to add or multiply the numbers? : ')

#if the user types add or addition, then print the sum of the two numbers
if choose == "add" or choose == "addition" : print("Sum: " + str(num1+num2))
 
#else if the user types multi or multiplication, then print the product of the two numbers
elif choose == "multi" or choose == "multiplication": print("Product: " + str(num1*num2))

else: print("sorry could not classify the input value")

confirm = input('Do you want to continue? type yes or no: ')
#if yes, then reload/continue
if confirm == "yes": os.execl(sys.executable, sys.executable, *sys.argv)

#else if: exit/quit
elif confirm == "no": quit()