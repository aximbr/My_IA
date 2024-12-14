"""
# Author: Jose P. Leitao
# 2024-12-13
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 5.01, 5.02, 5.03, 5.04
# Objective: load dataset from sklearn, use of Pandas and Multivariate regression
"""

# Import section
from os import system, name
from sklearn.datasets import fetch_openml
import pandas as pd

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


# Fetch the Boston housing dataset
# Since sklearn do not provide this dataset anymoree, is necessary use fetch from openml
boston_dataset = fetch_openml(name='boston', version=1)

# Print info from our Dataset
print(dir(boston_dataset))

# Print Description of Dataset
# print(boston_dataset.DESCR)

# Print Details of Dataset
# print(boston_dataset.details)

# Print Dataset data
# print(type(boston_dataset.data))

# Print the list of features
# print(boston_dataset.feature_names)

# Print the values of houses (target) in thousands (x1000 US$)
# print(boston_dataset.target)

# Create a dataframe from our dataset
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names )

# Add a column with prices
data['PRICE'] = boston_dataset.target

# Print the 5 first rows of our dataframe
print(data.head())

# Print the 5 last rows of our dataframe
print(data.tail())

# Print the number of instances (observations)
print(data.count())

## Cleaning the Data
# check for missing values
print(pd.isnull(data).any())