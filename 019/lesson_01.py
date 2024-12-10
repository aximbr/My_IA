"""
# Author: Jose P. Leitao
# 2024-12-10
# This program is used to trainning IA with Python
# based on "RealPython lesson - linear regression in Python" Course
# Lessons 0,1, 2
# Objective: simple linear regression
"""

# Import section
from os import system, name
import numpy as np
from sklearn.linear_model import LinearRegression

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

# Create some dummy points. It expect a 2D array with 6 row and 1 column
# after create the array we 'reshape' to fit our need
x = np.array([5,15, 25, 35, 45, 55]).reshape((-1,1))
# we could use (6,1) instead, but this a shortcut

# Our data points
y = np.array([5, 20, 14, 32, 22, 38])

# Let's print the shape for each array
print(x.shape)
print(y.shape)

# Let's create an object from class LinearRegression that will be our model
model=LinearRegression()

# Let's feed the data to this model
model.fit(x, y)

# Now our model contains the intercept and coeficient values:

print('The intercept is scalar', model.intercept_)
print('The coeficient is an array ', model.coef_)

# How good is our model? Check out using r square and input our observations
print('The R Square for our data points is ', model.score(x,y))

# Let's compare our prediction model with observed data
print('The actual values of y are ', y)
print('The predict values for y are ', model.predict(x))

# Let's create a new array from 0 to 5
x_new = np.arange(5).reshape((-1,1))
print('The new array is \n', x_new)

# Now lets evaluate y with this new x using our model
y_new = model.predict(x_new)
print('The new y with new x using our model are ', y_new)

