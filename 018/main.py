"""
# Author: Jose P. Leitao
# 2024-12-09
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 4.21, 4.22, 4.23
# Objective: MSE function, partial derivative and plotting the graphic
"""

# Import section
from os import system, name
import matplotlib as mplt
import matplotlib.pyplot as plt
import numpy as np

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

def mse(y, y_hat_in):
    """ MSE Function"""
    #return 1/y.size * sum((y-y_hat_in)**2)
    # necessary to include the index[0], otherwise np.average returns an Array
    # instead a scalar
    return np.average((y-y_hat_in)**2, axis=0)[0]

# Partial derivative of MSE wrt theta_0 is
def grad(x, y, thetas):
    """ x values, y values, thetas is array of theta parameters
    theta_0 at index 0, theta_1 at index 1"""
    n = y.size # number of values
    # Lets create an array to acomodate the partial derivative (slopes) wrt theta_0 and theta_1
    theta_0_slope = (-2/n)*sum(y - thetas[0] - thetas[1]*x)
    theta_1_slope = (-2/n)*sum((y - thetas[0] - thetas[1]*x)*x)
    # 3 way to return an array from our function
    #return np.array([theta_0_slope[0],theta_1_slope[0]])
    #return np.append(arr=theta_0_slope, values=theta_1_slope)
    return np.concatenate((theta_0_slope, theta_1_slope), axis=0)


# Main()
clear()

# Make data for thetas
NR_THETAS = 200

# Make sample data
# To use Linear Regression class, the Array should be on 2 dimension, it's necessary "reshape"
# the sample arrays.
# We can do this using transpose or reshape method
x_5 = np.array([[0.1, 1.2, 2.4, 3.2, 4.1, 5.7, 6.5]]).transpose()
y_5 = np.array([1.7, 2.4, 3.5, 3.0, 6.1, 9.4, 8.2]).reshape(7,1)

# Both methods give same result, a array with 7 rows and 1 column

# Show the shape of each array
#print('x_5 shape is: ', x_5.shape)
#print('y_5 shape is: ', y_5.shape)

# Define steps to learn
MULTIPLIER = 0.01

# Define our initial guess
thetas = np.array([2.9, 2.9])

# Variable to scatter plot (1 row, 2 columns)
plot_vals = thetas.reshape(1,2)
mse_vals = mse(y_5, thetas[0] + thetas[1]*x_5)

# Calculate new thetas 1000 times
for i in range(1000):
    thetas = thetas - MULTIPLIER * grad(x_5, y_5, thetas)
    # Append new values to plot_vals and mse_vals in 2 different ways
    plot_vals = np.concatenate((plot_vals, thetas.reshape(1,2)), axis=0)
    mse_vals = np.append(arr=mse_vals, values=mse(y_5, thetas[0] + thetas[1]*x_5))


# Print out results
# print('Min occurs at Theta_0 : ', thetas[0])
# print('Min occurs at Theta_1 : ', thetas[1])
# print('MSE is : ', mse(y_5, thetas[0] + thetas[1]*x_5))

# Create 2 1D arrays
th_0 = np.linspace(start=-1, stop=3, num=NR_THETAS)
th_1 = np.linspace(start=-1, stop=3, num=NR_THETAS)

# Create a tuple with 2D array
plot_t0, plot_t1 = np.meshgrid(th_0, th_1)

# print('Shape of th0 is ', th_0.shape)
# print('Shape for plot_t0 is ', plot_t0.shape)

# To calculate the MSE for each iteration of theta_0 with theta_1
# will use a nested loop
#
# create a 2D arrays of zeros with same number of rows and columns
plot_cost = np.zeros((NR_THETAS,NR_THETAS))
#print(plot_cost)

for i in range(NR_THETAS):
    for j in range(NR_THETAS):
        # calculate the predict value (y_hat) with out set of thetas
        y_hat = plot_t0[i][j] + plot_t1[i][j] * x_5
        # fill our plot array with MSE calculated from actual value and predicted value
        plot_cost[i][j] = mse(y_5, y_hat)

# print('Shape of plot_t0 is ', plot_t0.shape)
# print('Shape of plot_t1 is ', plot_t1.shape)
# print('Shape of plot_cost is ', plot_cost.shape)

# Plotting MSE
fig = plt.figure(figsize=[16,12])
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('Theta 0', fontsize=20)
ax.set_ylabel('Theta 1', fontsize=20)
ax.set_zlabel('Cost / MSE', fontsize=20)

ax.scatter(plot_vals[:,0],plot_vals[:,1],mse_vals, s=80, color='black' )
ax.plot_surface(plot_t0, plot_t1, plot_cost, cmap='rainbow', alpha=0.4)

# show the figure
plt.show()

