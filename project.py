
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


# ============== Importing data ======================
featureFile = 'combined_data.csv'
featureData = pd.read_csv(featureFile)
print(featureData)

labelFile = 'SyedHouse-GreenButton-New-ToJul2020.csv'
labelData = pd.read_csv(labelFile)
print(labelData)

# put feature and label in one object
data = pd.concat([featureData, labelData['Usage (kilowatt-hours)']], axis=1)
print(data)

# check feature-target correlation
# corr_matrix = data.corr()
# print(corr_matrix['Usage (kilowatt-hours)'].sort_values(ascending=False))

# heatmap
# sns.heatmap(corr_matrix, annot=True)
# plt.show()

plt.plot(data['Usage (kilowatt-hours)'])
plt.show()

# close up
plt.plot(data['Usage (kilowatt-hours)'].head(1000))
plt.show()

# TODO: data cleansing for feature data
print(data.isnull().sum())

# ================= Data Cleansing =======================

# drop the columns with mostly missing values
data = data.drop(columns=["Temp Flag", "Dew Point Temp Flag", "Rel Hum Flag", "Wind Dir Flag", "Wind Spd Flag",
                   "Visibility Flag", "Stn Press Flag", "Hmdx", "Hmdx Flag", "Wind Chill", "Wind Chill Flag"])

print(data.isnull().sum())

# Still several missing values in Temp, Dew point temp, rel hum, wind dir, wind spd, visibility, stn press
# and 8818 missing values in "weather"

# option 1 for handling a few missing values: get rid of the corresponding rows
# option 2 for handling a few missing values: replace empty fields with some value (zero, mean, median, etc.)

columns_to_clean = ["Temp (°C)", "Dew Point Temp (°C)", "Rel Hum (%)", "Wind Dir (10s deg)", "Wind Spd (km/h)",
                    "Visibility (km)", "Stn Press (kPa)"]

# option 1
# data.dropna(subset=columns_to_clean)

# option 2
# TODO: compute medians on entire set or strictly training set only? meeting to decide
columns_medians_dict = dict.fromkeys(columns_to_clean, 0)
for col in columns_to_clean:
    median = data[col].median()
    data[col].fillna(median, inplace=True)
    columns_medians_dict[col] = median

print(columns_medians_dict)

print(data.isnull().sum())





















