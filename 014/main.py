"""
# Author: Jose P. Leitao
# 2024-11-28
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 4.16
# Objective: Advance NumPy Array & 3D plot
"""

# Import section
from os import system, name
from math import log
import matplotlib as mplt
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm #Color Map
#List of Color Maps:
# https://matplotlib.org/stable/users/explain/colors/colormaps.html#colormaps

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

#print(params.shape)

# necessary to reshape the array to 1 row x 2 columns dimension
values_array = params.reshape(1,2)

#print(params.shape)
#print(values_array.shape)


for n in range(MAX_ITER):
    gradient_x = fpx(params[0], params[1])
    gradient_y = fpy(params[0], params[1])
    gradients = np.array([gradient_x, gradient_y])
    params = params - MULTIPLIER * gradients
    values_array = np.append(values_array, params.reshape(1,2), axis=0)
    #alternative to append
    #values_array = np.concatenate((values_array, params.reshape(1,2)), axis=0)

# Make Data for x and y
x_4 = np.linspace(start=-2, stop=2, num=200)
y_4 = np.linspace(start=-2, stop=2, num=200)

# Generate a 3D figure
fig = plt.figure(figsize=[16, 12])
ax = fig.add_subplot(projection='3d')

# convert to 2D Array
x_4, y_4 = np.meshgrid(x_4, y_4)

# Create our surface
# Set label to axis
ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y', fontsize=20)
ax.set_zlabel('f(x,y) - Cost', fontsize=20)

ax.plot_surface(x_4, y_4, f(x_4,y_4), cmap = 'Spectral', alpha=0.4)
ax.scatter(values_array[:,0], values_array[:,1],
           f(values_array[:,0], values_array[:,1]), s=50, color='red')

# show the figure
plt.show()



