"""
# Author: Jose P. Leitao
# 2024-11-27
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 4.14
# Objective: Batch Gradient Descent without Sympy
"""
# Reference for Derivative Calculation with symbols:
# https://www.symbolab.com/solver/derivative-calculator

# Import section
from os import system, name
from math import log
import numpy as np


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

# Partial Derivative Function wrt x:
def fpx(x,y):
    """Partial Derivative of f wrt x"""
    r = 3**(-x**2 - y**2)
    return 2*x*log(3)*r/(r +1)**2

# Partial Derivative Function wrt y:
def fpy(x,y):
    """Partial Derivative of f wrt y"""
    r = 3**(-x**2 - y**2)
    return 2*y*log(3)*r/(r +1)**2

# Main()
clear()

# Setup
MULTIPLIER = 0.1
MAX_ITER = 500
params = np.array([1.8, 1.0])  #initial guess

for n in range(MAX_ITER):
    gradient_x = fpx(params[0], params[1])
    gradient_y = fpy(params[0], params[1])
    gradients = np.array([gradient_x, gradient_y])
    params = params - MULTIPLIER * gradients

# Results
print('Values in gradient array ', gradients)
print('Minimun occurs at x value of: ', params[0])
print('Minimun occurs at y value of: ', params[1])
print('The Cost Function f(x,y) is ',f(params[0], params[1]))

