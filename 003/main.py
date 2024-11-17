"""
# Author: Jose P. Leitao
# 2024-11-17
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 3.9
# Objective: exercise with Python
"""

import pandas as pd
data = pd.read_csv('lsd_math_score_data.csv')

#print(data)
#print(data['Avg_Math_Test_Score'])

# Add a new column named Test_Subject and assigned the value 'Jennifer Lopez' to all rows
data['Test_Subject'] = 'Jennifer Lopez'
print(data)

# Add a new column named High_Score and assigned the value 100 to all rows
data['High_Score'] = 100
print(data)

# Chalenge: add 100 to value on Avg_Math_Test_Score and save on colummn High_Score
data['High_Score'] = data['Avg_Math_Test_Score'] + 100
print(data)

# Chalenge: square the value of High_Score
#data['High_Score'] = data['High_Score'] * data['High_Score']
#or
data['High_Score'] = data['High_Score'] **2
print(data)

# Chalenge: create a List called ColumnList and include 'LSD_ppm' and 'Avg_Math_Test_Score'
ColumnList = ['LSD_ppm', 'Avg_Math_Test_Score']

#Create a new dataframe with the values of columns above
cleandData = data[ColumnList]
print(cleandData)

#Instead to create a variable with list of columns, we can pass a list of desired columns
cleanData = data[['LSD_ppm', 'Avg_Math_Test_Score']]
print(cleandData)

#Explain difference between series and dataframe
y = data[['LSD_ppm']] #here we use a list to obtain a dataframe
print(y)
print(type(y))

z = data['LSD_ppm'] #here we use a string to obtain a Serie
print(z)
print(type(z))

# Chalenge: create a variable X type DataFrame that contains only the LSD_ppm
X = data[['LSD_ppm']]
print(X)
print(type(X))

#Remove the columm Test_Subject
del data['Test_Subject']
print(data)

# Chalenge: Remove the column High_Score
del data['High_Score']
print(data)
