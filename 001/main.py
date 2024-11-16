# Author: Jose P. Leitao
# 2024-11-16
# This program is use to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Objective: Load a csv file (gather and cleanning data were done) and display (explore)
# the data using pandas and matplotlib

import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt

data = pandas.read_csv('cost_revenue_clean.csv')

#print(data.describe())
X = DataFrame(data, columns=['production_budget_usd'])
y = DataFrame(data, columns=['worldwide_gross_usd'])

plt.figure(figsize=(10,6 ))
plt.scatter(X, y, alpha=0.3)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget US$')
plt.ylabel('Global Gross US$')
plt.ylim(0,3e09)
plt.xlim(0,4.5e08)
plt.show()
