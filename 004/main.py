"""
# Author: Jose P. Leitao
# 2024-11-20
# This program is used to trainning IA with Python
# based on "Complete Data Science & Machine Learning Bootcamp - Python 3" Course
# Lesson 3.18, 3.19
# Objective: exercise with Python
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('lsd_math_score_data.csv')

#extract each column as dataframes
time = data[['Time_Delay_in_Minutes']]
LSD = data[['LSD_ppm']]
score = data[['Avg_Math_Test_Score']]

#Plot a graphics that correlats time and LSD, using a "green" line
plt.title('Tissue Concentration of LSD over Time', fontsize=17)
plt.xlabel('Time in Minutes', fontsize=14)
plt.ylabel('Tissue LSD ppm', fontsize=14)
plt.text(x=30, y=2, s='Wagner et al. (1968)')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(1,7)
plt.xlim(0,500)

plt.style.use('classic')

plt.plot(time, LSD, color='g')
plt.show()

regr = LinearRegression()
regr.fit(LSD, score)
tetha1 = regr.coef_[0][0]
tetha0 = regr.intercept_[0]
R_Square = regr.score(LSD, score)

predict_score = regr.predict(LSD)

print(tetha0)
print(tetha1)
print(R_Square)

plt.scatter(LSD, score, color='blue', s=100, alpha = 0.7)
plt.title('Arithmetic vs LSD-25', fontsize=17)
plt.xlabel('Tissue LSD ppm', fontsize=14)
plt.ylabel('Performance Score', fontsize=14)
plt.ylim(25, 85)
plt.xlim(1, 6.5)

plt.plot(LSD, predict_score, color='red', linewidth=3)
plt.show()



