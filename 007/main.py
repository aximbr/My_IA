"""
# Author: Jose P. Leitao
# 2024-11-23
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 4.9
# Objective: Divergence, Overflow and Python Tuples
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


# Another cost function h(x)= X⁵ - 2x⁴ + 2
def h(x):
    """Cost function"""
    return x**5 - 2*x**4 + 2


# Define the derivative of h(x) as dh(x):
def dh(x):
    """Derivative function"""
    return 5*x**4 - 8*x**3

# Gradient Descend as Python Funtion
def gradient_descent(derivative_func, initial_guess, multiplier=0.02,
                     precision=0.001, max_iter=300):
    """
    Gradient Descent function
    default value for multiplier is 0.02
    default value for precision is 0.0001
    default value for max_iter is 300
    """
    new_x = initial_guess  # first guess
    x_list = [new_x]
    slope_list = [derivative_func(new_x)]

    for _ in range(max_iter):
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
x_3 = np.linspace(start=-2.5, stop=2.5, num=1000)

# Call the Gradient Descent function:
local_min, list_x, deriv_list = gradient_descent(derivative_func=dh, initial_guess=-0.2, max_iter=70)
# # Create a figure and put graphs together
plt.figure(figsize=[20, 5])

# #create the first graph (subplot)
plt.subplot(1, 2, 1)
plt.xlim(-1.2, 2.5)
plt.ylim(-1.0, 4.0)
plt.xlabel("x", fontsize=14)
plt.ylabel("h(x)", fontsize=14)
plt.title("Cost function")
plt.plot(x_3, h(x_3), color="blue", linewidth=3, alpha=0.8)
plt.scatter(list_x, h(np.array(list_x)), color='red', s=100, alpha=0.6)

# create the second graph (subplot)
plt.subplot(1, 2, 2)
plt.xlabel("x", fontsize=14)
plt.ylabel("dh(x)", fontsize=14)
plt.title("Slope of Cost function")
plt.xlim(-1.0, 2.0)
plt.ylim(-4.0, 5.0)
plt.grid()
plt.plot(x_3, dh(x_3), color="skyblue", linewidth=5, alpha=0.6)
plt.scatter(list_x, deriv_list, color='red', s=100, alpha=0.5)


# show the figure and subplots
plt.show()

print("The Local min occurs at ", local_min)
print('The cost of this min is: ', h(local_min))
print("The number of step: ", len(list_x))
