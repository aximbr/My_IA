"""
# Author: Jose P. Leitao
# 2024-11-30
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 4.19
# Objective: Implementing MSE function
"""

# Import section
from os import system, name
import matplotlib as mplt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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

def mse(y, y_hat):
    """ MSE Function"""
    #return 1/y.size * sum((y-y_hat)**2)
    return np.average((y-y_hat)**2, axis=0)

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

tetha_0 = regr.intercept_[0]
tetha_1 = regr.coef_[0][0]

# Show the intercept (theta 0) and coeficient angular (theta 1)
print('Theta 0 = ', tetha_0)
print('Theta 1 = ', tetha_1)

# y_hat = tetha_0 + tetha_1*X
y_hat = tetha_0 + tetha_1*x_5

print('The predict values are \n:', y_hat)
print('In comparision the actual values are \n', y_5)

# Time to show our MSE
print('The MSE calculated by our function is ', mse(y_5, y_hat))
print('The MSE calculated by sklearn function is', mean_squared_error(y_5, y_hat))
#Alternative
print('The MSE calculated by sklearn function is', mean_squared_error(y_5, regr.predict(x_5)))


# Lets plot the data and our function that predict the values
plt.scatter(x_5, y_5, s=50)
plt.plot(x_5, regr.predict(x_5), color='orange', linewidth=3)
plt.xlabel('X Values')
plt.ylabel('Y Values')

# show the figure
plt.show()


