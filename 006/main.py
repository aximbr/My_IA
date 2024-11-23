"""
# Author: Jose P. Leitao
# 2024-11-22
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 4.8
# Objective: Multiple Minima vs Initial Guess & Advanced Funtions
"""

# Import section
from os import system, name
import matplotlib as mplt
import matplotlib.pyplot as plt
import numpy as np

# Hide the toolbar when showing the figure
mplt.rcParams["toolbar"] = "None"


# define our clear function
def clear():
    """Clear screen"""
    # for windows
    if name == "nt":
        _ = system("cls")
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system("clear")


# Another cost funtion g(x)= X⁴ - 4x² + 5
def g(x):
    """Cost function"""
    return x**4 - 4 * x**2 + 5


# Define the derivative of g(x) as dg(x):
def dg(x):
    """Derivative function"""
    return 4 * x**3 - 8 * x

# Gradient Descend as Python Funtion
def gradient_descent(derivative_func, initial_guess, multiplier=0.02, precision=0.0001):
    """
    Gradient Descent function
    default value for multiplier is 0.02
    default value for precision is 0.0001
    """
    new_x = initial_guess  # first guess
    x_list = [new_x]
    slope_list = [derivative_func(new_x)]

    for n in range(500):
        previous_x = new_x
        gradient = derivative_func(previous_x)
        new_x = previous_x - multiplier * gradient

        x_list.append(new_x)
        slope_list.append(derivative_func(new_x))
        step_size = abs(new_x - previous_x)

        if step_size < precision:
            break
    return new_x, x_list, slope_list


# main()
clear()

# Make Data - generate an array of 1000 numbers starting at -2 and ending at 2
x_2 = np.linspace(start=-2, stop=2, num=1000)

# Call the Gradient Descent function:
local_min, list_x, deriv_list = gradient_descent(derivative_func=dg, initial_guess=0)
# # Create a figure and put graphs together
plt.figure(figsize=[20, 5])

# #create the first graph (subplot)
plt.subplot(1, 2, 1)
plt.xlim(-2, 2)
plt.ylim(0.5, 5.5)
plt.xlabel("x", fontsize=14)
plt.ylabel("g(x)", fontsize=14)
plt.title("Cost function")
plt.plot(x_2, g(x_2), color="blue", linewidth=3, alpha=0.8)
plt.scatter(list_x, g(np.array(list_x)), color='red', s=100, alpha=0.6)

# create the second graph (subplot)
plt.subplot(1, 2, 2)
plt.xlabel("x", fontsize=14)
plt.ylabel("dg(x)", fontsize=14)
plt.title("Slope of Cost function")
plt.xlim(-2, 3)
plt.ylim(-6, 8)
plt.grid()
plt.plot(x_2, dg(x_2), color="skyblue", linewidth=5, alpha=0.6)
plt.scatter(list_x, deriv_list, color='red', s=100, alpha=0.5)


# show the figure and subplots
plt.show()







print("The Local min occurs at ", local_min)
print("The number of step: ", len(list_x))


# # Create a figure and put both graph together
# plt.figure(figsize=[20,5])

# #create the first graph (subplot)
# plt.subplot(1,3,1)
# plt.xlim(-3,3)
# plt.ylim(0,8)
# plt.xlabel('x', fontsize=14)
# plt.ylabel('f(x)', fontsize=14)
# plt.title('Cost function')
# plt.plot(x_1, f(x_1), color='blue', linewidth=3, alpha=0.8)
# values = np.array(x_list)
# plt.scatter(x_list,f(values), color='red', s=100, alpha=0.6)

# #create the second graph (subplot)
# plt.subplot(1,3,2)
# plt.xlabel('x', fontsize=14)
# plt.ylabel('df(x)', fontsize=14)
# plt.title('Slope of Cost function')
# plt.xlim(-2,3)
# plt.ylim(-3,6)
# plt.grid()
# plt.plot(x_1, df(x_1), color='skyblue', linewidth=5, alpha=0.6)
# plt.scatter(x_list, slope_list, color='red', s=100, alpha=0.5)

# #create the third graph (subplot) - Derivative close-up
# plt.subplot(1,3,3)
# plt.xlabel('x', fontsize=14)
# plt.title('Gradient Descent (close-up)')
# plt.xlim(-0.55, -0.2)
# plt.ylim(-0.3, 0.8)
# plt.grid()
# plt.plot(x_1, df(x_1), color='skyblue', linewidth=6, alpha=0.8)
# plt.scatter(x_list, slope_list, color='red', s=300, alpha=0.5)
# #show the figure and subplots
# plt.show()
