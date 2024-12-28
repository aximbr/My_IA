"""
# Author: Jose P. Leitao
# 2024-12-14
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 5.05
# Objective: Visualisation Data - Histograms, Distributions and Bar Charts
"""

# Import section
from os import system, name
from sklearn.datasets import fetch_openml
import pandas as pd
import matplotlib as mplt
import matplotlib.pyplot as plt

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

# Main()
clear()


# Fetch the Boston housing dataset
# Since sklearn do not provide this dataset anymoree, is necessary use fetch from openml
boston_dataset = fetch_openml(name='boston', version=1)

# Create a dataframe from our dataset
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names )

# Add a column with prices
data['PRICE'] = boston_dataset.target

## Cleaning the Data
# check for missing values
#print(pd.isnull(data).any())

# Visualising Data prices
plt.figure(figsize=(10,6))
plt.hist(data["PRICE"], bins=50, edgecolor='black', color='#2196f3')
# Look on this website for color codes
# https://www.materialpalette.com/
plt.xlabel('Price of houses in 000s')
plt.ylabel('Number of Houses')
plt.show()
