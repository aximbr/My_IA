"""
# Author: Jose P. Leitao
# 2024-11-29
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 4.16
# Objective: Transposing and Reshaping Arrays
"""

# Import section
from os import system, name
import matplotlib as mplt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


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


# Main()
clear()

# Make sample data
# To use Linear Regression class, the Array should be on 2 dimension, it's necessary "reshape"
# the sample arrays.
# We can do this using transpose or reshape method
x_5 = np.array([[0.1, 1.2, 2.4, 3.2, 4.1, 5.7, 6.5]]).transpose()
y_5 = np.array([1.7, 2.4, 3.5, 3.0, 6.1, 9.4, 8.2]).reshape(7,1)

# Both methods give same result, a array with 7 rows and 1 column

# Show the shape of each array
print('x_5 shape is: ', x_5.shape)
print('y_5 shape is: ', y_5.shape)

# Quick Linear Regression
regr = LinearRegression()
regr.fit(x_5, y_5)

# Show the intercept (theta 0) and coeficient angular (theta 1)
print('Theta 0 = ', regr.intercept_[0])
print('Theta 1 = ', regr.coef_[0][0])

# Lets plot the data and our function that predict the values
plt.scatter(x_5, y_5, s=50)
plt.plot(x_5, regr.predict(x_5), color='orange', linewidth=3)
plt.xlabel('X Values')
plt.ylabel('Y Values')

# show the figure
plt.show()


