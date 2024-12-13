"""
# Author: Jose P. Leitao
# 2024-12-10
# This program is used to trainning IA with Python
# based on "RealPython lesson - linear regression in Python" Course
# Lessons 3, 4
# Objective: multiple linear regression
# Our X now is vector or array of n-Dimension
# Our observation is the form (X,y)
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

# Create some dummy points.
x = [[0,1], [5,1], [15,2], [25,5], [35,11], [45,15], [55,34], [60,35]]

# Our data points
y = [4, 5, 20, 14, 32, 22, 38, 43]

# Lets transform them on arrays

x = np.array(x)
y = np.array(y)

# Let's print the shape for each array
print(x.shape)
print(y.shape)

# Let's create an object from class LinearRegression that will be our model
model=LinearRegression()

# Let's feed the data to this model
model.fit(x, y)

# How good is our model? Check out using r square and input our observations
print('The R Square for our data points is ', model.score(x,y))

# Now our model contains the intercept and coeficient values:

print('The intercept is scalar', model.intercept_)
print('The coeficient is an array ', model.coef_)

# Let's compare our prediction model with observed data
print('The actual values of y are ', y)
print('The predict values for y are ', model.predict(x))

# Let's create a new array from 0 to 9
x_new = np.arange(10).reshape((-1,2))
print('The new array is \n', x_new)

# Now lets evaluate y with this new x using our model
y_new = model.predict(x_new)
print('The new y with new x using our model are ', y_new)

