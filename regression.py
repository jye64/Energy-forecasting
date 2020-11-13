
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


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

plt.plot(data['Usage (kilowatt-hours)'])
plt.show()

# close up
plt.plot(data['Usage (kilowatt-hours)'].head(200))
plt.show()

# ================= Data Cleansing =======================

# drop the columns with mostly missing values
data = data.drop(columns=["Temp Flag", "Dew Point Temp Flag", "Rel Hum Flag", "Wind Dir Flag", "Wind Spd Flag",
                          "Visibility Flag", "Stn Press Flag", "Hmdx", "Hmdx Flag", "Wind Chill", "Wind Chill Flag"])

# drop columns with constant data
data = data.drop(columns=["Longitude (x)", "Latitude (y)", "Station Name", "Climate ID", 'Date/Time'])

print(data.isnull().sum())

# Deal with missing values
columns_to_clean = ["Temp (°C)", "Dew Point Temp (°C)", "Rel Hum (%)", "Wind Dir (10s deg)", "Wind Spd (km/h)",
                    "Visibility (km)", "Stn Press (kPa)"]

# option 1: remove the corresponding rows
# data.dropna(subset=columns_to_clean)

# option 2: replace with zero, mean, or median
# TODO: compute medians on entire set or strictly training set only?
# columns_medians_dict = dict.fromkeys(columns_to_clean, 0)
# for col in columns_to_clean:
#     median = data[col].median()
#     data[col].fillna(median, inplace=True)
#     columns_medians_dict[col] = median
#
# print(columns_medians_dict)
#
# print(data.isnull().sum())

# option 3: replace with previous hour data
for col in columns_to_clean:
    data[col].fillna(method='ffill', inplace=True)
    data[col].fillna(method='bfill', inplace=True)

print(data.isnull().sum())

# clean the "weather" attribute: replace missing values with previous observations
data["Weather"].fillna(method='ffill', inplace=True)
data["Weather"].fillna(method='bfill', inplace=True)

print(data.isnull().sum())

# check feature-target correlation
corr_matrix = data.corr()
print(corr_matrix['Usage (kilowatt-hours)'].sort_values(ascending=False))

# heatmap
sns.heatmap(corr_matrix, annot=True)
plt.show()

sns.scatterplot(data=data, x="Temp (°C)", y="Usage (kilowatt-hours)")
plt.show()

data['Time'] = data['Time'].apply(lambda s: float(s.split(':')[0]))
print(data.dtypes)
print(data)

# TODO: Feature Engineering
season = []
for month in data['Month']:
    if 4 <= month <= 5:
        season.append('Spring')
    elif 6 <= month <= 9:
        season.append('Summer')
    elif 10 <= month <= 11:
        season.append('Fall')
    else:
        season.append('Winter')

season = pd.DataFrame(data=season, columns=['Season'])
print(season)

data = pd.concat([data, season], axis=1)
print(data)

data_onehot = pd.get_dummies(data, columns=['Season', 'Weather'])
print(data_onehot)

X = data_onehot.drop(columns=['Usage (kilowatt-hours)'], axis=1)
Y = data_onehot['Usage (kilowatt-hours)']

# split training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# standard Scaler to fit training data in X
scalerX = StandardScaler().fit(X_train)
X_train_std = scalerX.transform(X_train)
X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)

# apply the same scaler on testing data
X_test_std = scalerX.transform(X_test)
X_test_std = pd.DataFrame(X_test_std, columns=X_test.columns)


# ===================== Part 3: Modeling =====================

models_to_evaluate = [DecisionTreeRegressor(), RandomForestRegressor()]

for regr in models_to_evaluate:
    model_name = str(regr)
    regr.fit(X_train_std, Y_train)

    # copy the DataFrame indexes
    results = X_train.copy()
    results["predicted"] = regr.predict(X_train_std)
    results["actual"] = Y_train
    results = results[['predicted', 'actual']]
    results['predicted'] = results['predicted'].round(2)

    # reset the index of DataFrame and use the default indexing (0 1 2 3...N-1)
    results = pd.DataFrame.reset_index(results, drop=True)

    # visualize predicted vs actual in train set
    plt.plot(results['predicted'].head(100), label='predicted')
    plt.plot(results['actual'].head(100), label='actual')
    plt.xlabel('index in train set')
    plt.ylabel('price')
    plt.title(model_name + ':Predicted vs Actual in Train set')
    plt.legend()
    plt.show()


# ===================== Part 4: Accuracy & Evaluation =====================

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


model_performance = pd.DataFrame(columns=['Model', 'Train MAE', 'Train RMSE', 'Train MAPE',
                                          'CV RMSE', 'Test MAE', 'Test RMSE', 'Test MAPE'])

for regr in models_to_evaluate:
    model_name = str(regr)

    Y_train_pred = regr.predict(X_train_std)

    Train_MAE = mean_absolute_error(Y_train, Y_train_pred).round(2)
    Train_RMSE = np.sqrt(mean_squared_error(Y_train, Y_train_pred).round(2))
    Train_MAPE = mean_absolute_percentage_error(Y_train, Y_train_pred).round(2)

    # 5 - fold Cross Validation on training data for model validation
    # RMSE
    CV = cross_validate(regr, X_train_std, Y_train, cv=5, scoring='neg_root_mean_squared_error')
    CV['test_score'] = -CV['test_score']
    CV_Overall_RMSE = np.mean(CV['test_score']).round(2)

    # after validating the model, use the test set to compute generalization error
    Y_test_pred = regr.predict(X_test_std)

    Test_MAE = mean_absolute_error(Y_test, Y_test_pred).round(2)
    Test_RMSE = np.sqrt(mean_squared_error(Y_test, Y_test_pred)).round(2)
    Test_MAPE = mean_absolute_percentage_error(Y_test, Y_test_pred).round(2)

    model_performance = model_performance.append({'Model': model_name, 'Train MAE': Train_MAE,
                                                  'Train RMSE': Train_RMSE, 'Train MAPE': Train_MAPE,
                                                  'CV RMSE': CV_Overall_RMSE, 'Test MAE': Test_MAE,
                                                  'Test RMSE': Test_RMSE, 'Test MAPE': Test_MAPE},
                                                 ignore_index=True)

    # copy the DataFrame indexes
    test_results = X_test.copy()
    test_results["predicted"] = Y_test_pred
    test_results["actual"] = Y_test
    test_results = test_results[['predicted', 'actual']]
    test_results['predicted'] = test_results['predicted'].round(2)

    # reset the index of DataFrame and use the default indexing (0 1 2 3...N-1)
    test_results = pd.DataFrame.reset_index(test_results, drop=True)

    # visualize predicted vs actual in test set
    plt.plot(test_results['predicted'].head(100), label='predicted')
    plt.plot(test_results['actual'].head(100), label='actual')
    plt.xlabel('index in test set')
    plt.ylabel('price')
    plt.title(model_name + ':Predicted vs Actual in test set')
    plt.legend()
    plt.show()

pd.set_option('max_columns', 8)
print(model_performance)
