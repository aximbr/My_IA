"""This is a test program
JPL - 2024-12-25
V 0.1
"""
# Import section
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression

# Hide the toolbar when showing the figure
mplt.rcParams["toolbar"] = "None"

# Our cost function MSE
def mse(y, y_hat_in):
    """ MSE Function"""
    # necessary to include the index[0], otherwise np.average returns an Array
    # instead a scalar
    return np.average((y-y_hat_in)**2, axis=0)[0]

# Partial derivative of MSE wrt theta_0 and theta_1
def grad(x, y, thetas):
    """ x values, y values, thetas is array of theta parameters
    theta_0 at index 0, theta_1 at index 1"""
    n = y.size # number of values
    # Lets create an array to acomodate the partial derivative (slopes) wrt theta_0 and theta_1
    theta_0_slope = (-2/n)*sum(y - thetas[0] - thetas[1]*x)
    theta_1_slope = (-2/n)*sum((y - thetas[0] - thetas[1]*x)*x)
    return np.concatenate((theta_0_slope, theta_1_slope), axis=0)

# Main()

# Create a dummy dataset
NUM_DATAPOINTS = 20
x_real = np.linspace(start=0.5, stop=8.6, num=NUM_DATAPOINTS).reshape(NUM_DATAPOINTS,1)
y_real = np.random.rand(NUM_DATAPOINTS, 1)
y_real = 1.5 + 2.5*y_real


# Create a Linear Regression Model using our Dataset
# regr = LinearRegression()

# regr.fit(x_real,y_real)

# tetha_0 = regr.intercept_[0]
# tetha_1 = regr.coef_[0][0]
# y_hat = regr.predict(x_real)

# Make data for thetas
NR_THETAS = 200
# Define steps to learn
MULTIPLIER = 0.01

# Define our initial guess
thetas = np.array([0, 0])

# Variable to plot thetas X MSE
plot_vals = thetas.reshape(1,2)
mse_vals = mse(y_real, thetas[0] + thetas[1]*x_real)   # y_hat = theta_0 + theta_1 * x

# Calculate new thetas 1000 times
for i in range(1000):
    thetas = thetas - MULTIPLIER * grad(x_real, y_real, thetas)
    # Append new values to plot_vals and mse_vals in 2 different ways
    plot_vals = np.concatenate((plot_vals, thetas.reshape(1,2)), axis=0)
    mse_vals = np.append(arr=mse_vals, values=mse(y_real, thetas[0] + thetas[1]*x_real))


# Plot the datapoint with linear regression and MSE for multiple thetas
# # Create a figure and put graphs together
fig = plt.figure(figsize=[16,12])

# Create 2 1D arrays
th_0 = np.linspace(start=-1, stop=3, num=NR_THETAS)
th_1 = np.linspace(start=-1, stop=3, num=NR_THETAS)

# Create a tuple with 2D array
plot_t0, plot_t1 = np.meshgrid(th_0, th_1)

# create a 2D arrays of zeros with same number of rows and columns
plot_cost = np.zeros((NR_THETAS,NR_THETAS))

for i in range(NR_THETAS):
    for j in range(NR_THETAS):
        # calculate the predict value (y_hat) with out set of thetas
        y_hat = plot_t0[i][j] + plot_t1[i][j] * x_real
        # fill our plot array with MSE calculated from actual value and predicted value
        plot_cost[i][j] = mse(y_real, y_hat)


y_hat = thetas[0] + thetas[1]* x_real

# #create the first graph (subplot)
plt.subplot(1, 2, 1)
plt.xlim(0, 10)
plt.ylim(0, 5)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.title("My Dataset")
plt.scatter(x_real, y_real, color='blue', s=100, alpha=0.6)
plt.plot(x_real, y_hat, color='red')

# create the second graph (subplot)
ax = fig.add_subplot(1, 2, 2, projection='3d')


ax.set_xlabel('Theta 0', fontsize=20)
ax.set_ylabel('Theta 1', fontsize=20)
ax.set_zlabel('Cost / MSE', fontsize=20)

# plot the surficie
ax.plot_surface(plot_t0, plot_t1, plot_cost, cmap='rainbow', alpha=0.4)

# plot the thetas until minimize our cost function
ax.scatter(plot_vals[:,0],plot_vals[:,1],mse_vals, s=50, color='black' )


# show the figure and subplots
plt.show()
