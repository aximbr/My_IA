"""
# Author: Jose P. Leitao
# 2024-11-26
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 4.12
# Objective: Partial Derivative & Symbolic Computation
"""
# Reference for Derivative Calculation with symbols:
# https://www.symbolab.com/solver/derivative-calculator

# Import section
from os import system, name
from sympy import symbols, diff

# define our clear function
def clear():
    """Clear screen"""
    # for windows
    if name == "nt":
        _ = system("cls")
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system("clear")

# Cost funtion f(x,y)= 1/(3^(-x² -y²) + 1)
def f(x,y):
    """Cost function"""
    r = 3**(-x**2 - y**2)
    return 1/(r + 1)

# Main()
clear()

a, b = symbols('x, y')

print('Our Cost Function f(x,y) is ',f(a,b))

print('Partial Derivative with respect to x is ', diff(f(a,b), a))

print('Value of f(x,y) when x=1.8 and y=1.0 is',
                                f(a,b).evalf(subs={a:1.8, b:1.0}))
                                
print('Value of Partial Derivative wrt x, when x=1.8 and y=1.0 is',
                                diff(f(a,b),a).evalf(subs={a:1.8, b:1.0}))

