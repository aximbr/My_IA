"""
# Author: Jose P. Leitao
# 2024-11-28
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 4.15
# Objective: Advance NumPy Array
"""

# Import section
from os import system, name
import numpy as np


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

# Create a basic Array (1 row with 2 columns)
kirk = np.array([['Captain', 'Guitar']])
# Show the shape of Array
print(kirk.shape)

# Create another Array (2 rows with 2 columns)
hs_band = np.array([['Black Thought', 'MC'], ['QuestLove', 'Drums']])
# Show the shape of new Array
print(hs_band.shape)

# Print just one element of Second Array
print('hs_band[0][1] : ', hs_band[0][1])

# Print the whole row
print('hs_band[0] : ',hs_band[0])

# Create an Array the is thea append of one array
# arr=Array origen, values=what we want to append, axis=dimension(0=row, 1=column)
the_roots = np.append(arr=hs_band, values=kirk, axis=0)
print('the new array is \n',the_roots)

# Slicing an Array
# Selecting all elements of first column
print('The nicknames are ...\n', the_roots[:,0])

# Selecting all elements of second column
print('The roles are ...\n', the_roots[:,1])

# Append a new element to this Array
the_roots = np.append(arr=the_roots, values=[['Malik B', 'MC']], axis=0)
print('The Array after append is \n', the_roots)


