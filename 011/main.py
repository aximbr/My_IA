"""
# Author: Jose P. Leitao
# 2024-11-26
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 4.13
# Objective: Batch Gradient Descent with Sympy
"""
# Reference for Derivative Calculation with symbols:
# https://www.symbolab.com/solver/derivative-calculator

# Import section
from os import system, name
import numpy as np
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

# Setup
a, b = symbols('x, y')

MULTIPLIER = 0.1
MAX_ITER = 200
params = np.array([1.8, 1.0])  #initial guess

for n in range(MAX_ITER):
    gradient_x = diff(f(a,b), a).evalf(subs={a:params[0], b:params[1]})
    gradient_y = diff(f(a,b), b).evalf(subs={a:params[0], b:params[1]})
    gradients = np.array([gradient_x, gradient_y])
    params = params - MULTIPLIER * gradients

# Results
print('Values in gradient array ', gradients)
print('Minimun occurs at x value of: ', params[0])
print('Minimun occurs at y value of: ', params[1])
print('The Cost Function f(x,y) is ',f(params[0], params[1]))
