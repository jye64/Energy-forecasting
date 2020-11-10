
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


# ============== Importing data ======================
featureDataFile = 'combined_data.csv'
featureData = pd.read_csv(featureDataFile)
print(featureData)

labelFile = 'SyedHouse-GreenButton-New-ToJul2020.csv'
labelData = pd.read_csv(labelFile)
print(labelData)

# TODO 1: should delete the first line in the label file, otherwise key is unmatched
# TODO 2: fix time period mismatch, i.e. should be between 2018.7.7 00:00 - 2020.7.6 11:00, manually delete in data
data = pd.concat([featureData, labelData['Usage (kilowatt-hours)']], axis=1)
print(data)

# check feature-target correlation
# corr_matrix = data.corr()
# print(corr_matrix['Usage (kilowatt-hours)'].sort_values(ascending=False))

# heatmap
# sns.heatmap(corr_matrix, annot=True)
# plt.show()

plt.plot(labelData['Usage (kilowatt-hours)'])
plt.show()

# close up
plt.plot(labelData['Usage (kilowatt-hours)'].head(1000))
plt.show()









