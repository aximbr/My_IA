"""
# Author: Jose P. Leitao
# 2024-11-23
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 4.10
# Objective: Learning Rate
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
x_2 = np.linspace(start=-2, stop=2, num=1000)

# Call the Gradient Descent function with different learn rate
n = 100
low_gama = gradient_descent(derivative_func=dg,
                            initial_guess=3, 
                            multiplier=0.0005,
                            precision=0.0001,
                            max_iter=n)

mid_gama = gradient_descent(derivative_func=dg,
                            initial_guess=3, 
                            multiplier=0.001,
                            precision=0.0001,
                            max_iter=n)

high_gama = gradient_descent(derivative_func=dg,
                            initial_guess=3, 
                            multiplier=0.002,
                            precision=0.0002,
                            max_iter=n)

#insane learning rate
insane_gama = gradient_descent(derivative_func=dg,
                            initial_guess=1.9, 
                            multiplier=0.25,
                            precision=0.0002,
                            max_iter=n)

# Plot reduction in cost for each iteration
plt.figure(figsize=[20, 10])

plt.xlim(0, n)
plt.ylim(0, 50)
plt.xlabel("Nr. of iterations", fontsize=14)
plt.ylabel("Cost", fontsize=14)
plt.title("Effect of Learning Rate")

# Values for our chart
# 1) Y axis data : Convert the list into Array

low_values = np.array(low_gama[1])

# 2) X axis data: create a list from 0 to n+1
iteration_list = list(range(0, n+1))

# Plot low learning rate
plt.plot(iteration_list, g(low_values), color="lightgreen", linewidth=5)
plt.scatter(iteration_list, g(low_values), color='lightgreen', s=80, alpha=0.6)

# Plot mid learning rate
plt.plot(iteration_list, g(np.array(mid_gama[1])), color="steelblue", linewidth=5)
plt.scatter(iteration_list, g(np.array(mid_gama[1])), color='steelblue', s=80, alpha=0.6)

# Plot high learning rate
plt.plot(iteration_list, g(np.array(high_gama[1])), color="hotpink", linewidth=5)
plt.scatter(iteration_list, g(np.array(high_gama[1])), color='hotpink', s=80, alpha=0.6)

# Plot insane learning rate
plt.plot(iteration_list, g(np.array(insane_gama[1])), color="red", linewidth=5)
plt.scatter(iteration_list, g(np.array(insane_gama[1])), color='red', s=80, alpha=0.6)


# show the figure
plt.show()

# print("The Local min occurs at ", low_gama[0])
# print('The cost of this min is: ', g(low_gama[0]))
# print("The number of step: ", len(low_gama[1]))
