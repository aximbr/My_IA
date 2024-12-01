"""
# Author: Jose P. Leitao
# 2024-12-01
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 4.20
# Objective: Nesting loops
"""

# Import section
from os import system, name
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

# Main()
clear()

# Make data for thetas
NR_THETAS = 5

# Create 2 1D arrays
th_0 = np.linspace(start=-1, stop=3, num=NR_THETAS)
th_1 = np.linspace(start=-1, stop=3, num=NR_THETAS)

# Create a tuple with 2D array
plot_t0, plot_t1 = np.meshgrid(th_0, th_1)

print('Shape of th0 is ', th_0.shape)
print('Shape for plot_t0 is ', plot_t0.shape)

# To calculate the MSE for each iteration of theta_0 with theta_1
# will use a nested loop
#
# create a 2D arrays of zeros with same number of rows and columns
plot_cost = np.zeros((NR_THETAS,NR_THETAS))
print(plot_cost)

# practice with nested loop
for i in range(3):
    for j in range(3):
        print(f'The value of i is {i}, and j is {j}')

