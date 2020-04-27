#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2 17:36:21 2020

@author: arushimadan
"""

import os
print(os.listdir('../Desktop'))


# Importing necessary libraries
import pandas as pd
import seaborn as sns
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Importing the dataset
path_data = os.path.join('..', 'Desktop', "IDPDataset.csv")
tides = pd.read_csv('../Desktop/IDPDataset.csv')

# Breakdown of datetime into date, month, year and time
tides.columns = ["datetime", "Height", "Residual", "Total Height"]
tides["datetime"] = pd.to_datetime(tides["datetime"])
tides["year"] = tides["datetime"].astype(str).str[0:4]
#tides["year"] = tides["year"].astype(int)

tides["time1"] = tides["datetime"].astype(str).str[11:16]
tides["month"] = tides["datetime"].astype(str).str[5:7]
tides["date"] = tides["datetime"].astype(str).str[8:10]

# Verify how dataset looks
tides.info()
tides.head()
tides.shape


plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(tides['Total Height'])

## Data Visualization
# tides.pivot_table('Height', index="time1").plot(figsize=(20,10))
# tides.pivot_table('Residual', index="time1").plot(figsize=(20,10))
# tides.pivot_table('Total Height', index="time1").plot(figsize=(20,10))

## Total height behaviour of each month performed at certain times as Height vs time graphs
# tides[tides.month == "01"].pivot_table('Total Height', index="time1").plot(figsize=(20,10))
# tides[tides.month == "02"].pivot_table('Total Height', index="time1").plot(figsize=(20,10))
# tides[tides.month == "03"].pivot_table('Total Height', index="time1").plot(figsize=(20,10))
# tides[tides.month == "04"].pivot_table('Total Height', index="time1").plot(figsize=(20,10))
# tides[tides.month == "05"].pivot_table('Total Height', index="time1").plot(figsize=(20,10))
# tides[tides.month == "06"].pivot_table('Total Height', index="time1").plot(figsize=(20,10))
# tides[tides.month == "07"].pivot_table('Total Height', index="time1").plot(figsize=(20,10))
# tides[tides.month == "08"].pivot_table('Total Height', index="time1").plot(figsize=(20,10))
# tides[tides.month == "09"].pivot_table('Total Height', index="time1").plot(figsize=(20,10))
# tides[tides.month == "10"].pivot_table('Total Height', index="time1").plot(figsize=(20,10))
# tides[tides.month == "11"].pivot_table('Total Height', index="time1").plot(figsize=(20,10))
# tides[tides.month == "12"].pivot_table('Total Height', index="time1").plot(figsize=(20,10))

## Height vs date graphs for each month
# tides.pivot_table('Total Height', index='date', columns='month').plot()

## Height vs month for each year
#tides.pivot_table('Total Height', index='month', columns='year').plot()

# Plotting a pair-plot to confirm obvious assumptions.
g = sns.pairplot(tides)

# dropping axes not needed as inputs. We only need 'datetime' as that
# combines date and time and is in the datetime64 format.
x_tides = tides.drop("time1",axis = 1)
x_tides = x_tides.drop("Total Height",axis = 1)
x_tides = x_tides.drop("Height",axis = 1)
x_tides = x_tides.drop("Residual",axis = 1)
x_tides = x_tides.drop("month",axis = 1)
x_tides = x_tides.drop("date",axis = 1)
x_tides = x_tides.drop("year",axis = 1)


# checking rows and columns
print(x_tides.shape)
y = tides["Total Height"]
print(y.shape)
 

# Splitting the data into train and test data
x_tides_train, x_tides_test, y_train, y_test = train_test_split(x_tides, y, test_size=0.2, random_state=0)
                
# Import Random Forest regressor
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 128 decision trees. The number of trees was 
# figured out by a n_estimators vs accuracy graph, the optimum number of 128 was evaluated.
rf = RandomForestRegressor(n_estimators = 128, random_state = 0)

# Train the model on training data
rf.fit(x_tides_train, y_train);

# Use the forest's predict method on the test data
predictions = rf.predict(x_tides_test)

# Bar chart to show prediction accuracy of 10 randomly chosen values.
df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
df1 = df.head(10)
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.suptitle('Comparison of predicted vs real values', fontsize=20)
plt.xlabel('Row number in dataset corresponding to date and time', fontsize=16)
plt.ylabel('Total Height', fontsize=16)
plt.show()

# Calculate the absolute errors
errors = abs(predictions - y_test)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

