"""
# Author: Jose P. Leitao
# 2024-11-21
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 4.4 - 4.6
# Objective: Gradiente Descent
"""

import matplotlib as mplt
import matplotlib.pyplot as plt
import numpy as np

mplt.rcParams['toolbar'] = 'None'

# A simple cost funtion f(x)= X² + x + 1
def f(x):
    """ Cost function example"""
    return x**2 + x + 1

# Define the derivative of f(x) as df(x):
def df(x):
    """Derivative function"""
    return 2*x + 1

# Make Data - generate an array of 100 numbers starting at -3 and ending at 3
x_1 = np.linspace(start=-3, stop=3, num=100)

#Gradient Descend
new_x = 3 #first guess
previous_x = 0 #just to initiate the variable
STEP_MULTIPLIER = 0.1 #the step
PRECISION = 0.0001

x_list = [new_x]
slope_list = [df(new_x)]

for n in range(500):
    previous_x = new_x
    gradient = df(previous_x)
    new_x = previous_x - STEP_MULTIPLIER * gradient

    x_list.append(new_x)
    slope_list.append(df(new_x))
    step_size = abs(new_x - previous_x)

    if step_size < PRECISION :
        print('The loop ran ', n, ' times')
        break

print('Local minimun occurs at ', new_x)
print('Slope or df(x) value at this point is ', df(new_x))
print('f(x) or cost function at this point is ', f(new_x))

# Create a figure and put both graph together
plt.figure(figsize=[20,5])

#create the first graph (subplot)
plt.subplot(1,3,1)
plt.xlim(-3,3)
plt.ylim(0,8)
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)
plt.title('Cost function')
plt.plot(x_1, f(x_1), color='blue', linewidth=3, alpha=0.8)
values = np.array(x_list)
plt.scatter(x_list,f(values), color='red', s=100, alpha=0.6)

#create the second graph (subplot)
plt.subplot(1,3,2)
plt.xlabel('x', fontsize=14)
plt.ylabel('df(x)', fontsize=14)
plt.title('Slope of Cost function')
plt.xlim(-2,3)
plt.ylim(-3,6)
plt.grid()
plt.plot(x_1, df(x_1), color='skyblue', linewidth=5, alpha=0.6)
plt.scatter(x_list, slope_list, color='red', s=100, alpha=0.5)

#create the third graph (subplot) - Derivative close-up
plt.subplot(1,3,3)
plt.xlabel('x', fontsize=14)
plt.title('Gradient Descent (close-up)')
plt.xlim(-0.55, -0.2)
plt.ylim(-0.3, 0.8)
plt.grid()
plt.plot(x_1, df(x_1), color='skyblue', linewidth=6, alpha=0.8)
plt.scatter(x_list, slope_list, color='red', s=300, alpha=0.5)
#show the figure and subplots
plt.show()





