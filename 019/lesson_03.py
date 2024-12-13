"""
# Author: Jose P. Leitao
# 2024-12-11
# This program is used to trainning IA with Python
# based on "RealPython lesson - linear regression in Python" Course
# Lessons 5, 6, 7
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



# 1 - read/import/create observation data
# 2 - make them on array format
# 3 - generate the transform for X observation (x_)
# 4 - create a object of class LinearRegression (model)
# 5 - fit the model with transform (x_) and data result (y)
# 6 - check the R Square to verify how good is our model
# 7 - find the coeficients b0, b1, b2, ...
# 8 - compare our model (predict values) against actual values (y)
# 9 - do some predictions with new X values

# Main()
# clear()

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

# Now let's assume x is 2D form
x = [[0,1], [5,1], [15,2], [25,5], [35,11], [45,15], [55,34], [60,35]]

# Our data points
y = [4, 5, 20, 14, 32, 22, 38, 43]

x = np.array(x)
y = np.array(y)

x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

print(x_)
# For each row we have the first and second term of observation,
# the theird is the first term to square, 
# the forth is the first term x second term
# and the last element of the row is second term to square

# # Let's create an object from class LinearRegression that will be our model
model=LinearRegression()

# Let's feed the data to this model
model.fit(x_, y)

# How good is our model? Check out using r square and input our observations
print('The new R Square for our data points is ', model.score(x_,y))

# Now our new model contains the intercept and coeficient values:
# b0
print('The intercept is scalar', model.intercept_)
# b1, b2 for x1 and x2
# b3 for x1²
# b4 for x1x2
# b5 for x2²
print('The coeficient is an array ', model.coef_)
