"""
# Author: Jose P. Leitao
# 2024-12-11
# This program is used to trainning IA with Python
# based on "RealPython lesson - linear regression in Python" Course
# Lessons 5, 6
# Objective: Polynomial Regression
"""

# Import section
from os import system, name
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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
x = [5, 15, 25, 35, 45, 55]

# Our data points
y = [15, 11, 2, 8, 25, 32]

# Lets transform them on arrays

x = np.array(x).reshape(-1,1)
y = np.array(y)

# Let's print the shape for each array
print(x.shape)
print(y.shape)
print(x)

# Let's create our transformer without the '1' column
transformer = PolynomialFeatures(degree=2, include_bias=False)
x_ = transformer.fit_transform(x)

print(x_)


# # Let's create an object from class LinearRegression that will be our model
model=LinearRegression()

# Let's feed the data to this model
model.fit(x_, y)

# How good is our model? Check out using r square and input our observations
print('The R Square for our data points is ', model.score(x_,y))

# Now our model contains the intercept and coeficient values:

print('The intercept is scalar', model.intercept_)
print('The coeficient is an array ', model.coef_)

# Let's compare our prediction model with observed data
print('The actual values of y are ', y)
print('The predict values for y are ', model.predict(x_))
print('The difference between y and predict_y is ', y - model.predict(x_))
