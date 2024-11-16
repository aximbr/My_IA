"""
# Author: Jose P. Leitao
# 2024-11-16
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Objective: Load a csv file and obtain the parameter teta_0 and teta_1 for our
# Linear regression
"""

import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pandas.read_csv('cost_revenue_clean.csv')

#print(data.describe())
X = DataFrame(data, columns=['production_budget_usd'])
y = DataFrame(data, columns=['worldwide_gross_usd'])



# Calculate the linear regression
regression = LinearRegression()
regression.fit(X, y)

# Slope Coeficient (teta_1)
#print(regression.coef_)

# Intercept (teta_0)
#print(regression.intercept_)

#plot graphics
plt.figure(figsize=(10,6 ))
plt.scatter(X, y, alpha=0.3)
#include the Linear Regression line
plt.plot(X, regression.predict(X), color='red', linewidth=4)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget US$')
plt.ylabel('Global Gross US$')
plt.ylim(0,3e09)
plt.xlim(0,4.5e08)
plt.show()

#Calculate our score (Goodness of fit)
print(regression.score(X, y))

#This value represent the amount of right prediction vs actual values
