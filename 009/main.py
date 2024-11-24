"""
# Author: Jose P. Leitao
# 2024-11-24
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 4.11
# Objective: Data Vizualisation with 3D charts
"""

# Import section
from os import system, name
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


# main()
clear()

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

ax.plot_surface(x_4, y_4, f(x_4,y_4), cmap = cm.Spectral, alpha=0.4)

# show the figure
plt.show()

