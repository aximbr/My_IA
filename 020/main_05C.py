"""
# Author: Jose P. Leitao
# 2025-01-01
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Objective: Generic template for Multivariable Regression
"""

# Import section
from os import system, name

from sklearn.datasets import fetch_openml
# due changes on sci-learn version the example dataset for Boston House prices
# is not available, but you can download from openml site using fecth_openml module
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np

# Hide the toolbar when showing the figure
#mplt.rcParams["toolbar"] = "None"

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

## Obtain data

# Fetch the Boston housing dataset
# Since sklearn do not provide this dataset anymoree, is necessary use fetch from openml
boston_dataset = fetch_openml(name='boston', version=1)

# Create a dataframe from our dataset
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names )

# Add a column with prices = target
data['PRICE'] = boston_dataset.target

## Verify data

# check for missing values
print(pd.isnull(data).any())

# Info about data
print(data.info())
print(data.describe())

# Correct data type for some columns
data['CHAS'] = data['CHAS'].astype('float64')
data['RAD'] = data['RAD'].astype('float64')
print(data.info())

## verify correlation between features and target
print(data.corr())

## split features and target
target = data['PRICE']
features = data.drop('PRICE', axis=1)

## Create a train and test set
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

## Regression Model
regr = LinearRegression()

## feed tte model with train data
regr.fit(X_train, y_train)

## List the coeficients (Thetas)
print('Theta 0:', regr.intercept_)
print('Theta 1 to n:', pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['Coef']))

## Verify the R2 of model
print('r-squared for Data Training:', regr.score(X_train, y_train))
print('r-squared for Data Test', regr.score(X_test, y_test))



