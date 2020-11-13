import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# ================= Configure GPU ====================
config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


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

data = data[data.Season != 'Winter']
print(data)

data_onehot = pd.get_dummies(data, columns=['Season', 'Weather'])
print(data_onehot)

X = data_onehot.drop(columns=['Usage (kilowatt-hours)'], axis=1)
Y = data_onehot[['Usage (kilowatt-hours)']]

# split training, validation, test set as 60% : 20% : 20%
X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_full, Y_train_full, test_size=0.25, random_state=42)

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)

# standardization
scalerX = StandardScaler()
X_train_scaled = scalerX.fit_transform(X_train)
X_valid_scaled = scalerX.transform(X_valid)
X_test_scaled = scalerX.transform(X_test)

scalerY = StandardScaler()
Y_train_scaled = scalerY.fit_transform(Y_train)
Y_valid_scaled = scalerY.transform(Y_valid)
Y_test_scaled = scalerY.transform(Y_test)

# ========================= Modeling ==============================

# input_shape = X_train.shape[1:]


# def build_model(n_hidden=1, n_neurons=112, learning_rate=0.01, init='glorot_uniform', activation='relu'):
#     model = keras.models.Sequential()

#     # input layer
#     model.add(layers.Dense(112, activation=activation, kernel_initializer=init, input_shape=input_shape))

#     # hidden layers
#     for layer in range(n_hidden):
#         model.add(layers.Dense(n_neurons, activation=activation, kernel_initializer=init))

#     # output layer
#     model.add(layers.Dense(1, kernel_initializer=init))
#     optimizer = keras.optimizers.SGD(learning_rate)
#     model.compile(loss='mse', optimizer=optimizer)
#     return model


# keras_reg = KerasRegressor(build_model)


# # ==================== Hyperparameters Tuning ===================

# hidden_layers = [5, 6, 7, 8]
# neurons = list(range(20, 100))
# learn_rate = [0.01, 0.001, 0.002]
# init_mode = ['he_normal', 'random_normal', 'glorot_normal']
# activate = ['relu', 'elu']


# param_grid = dict(n_hidden=hidden_layers, n_neurons=neurons, learning_rate=learn_rate,
#                   init=init_mode, activation=activate)

# rnd_search_cv = RandomizedSearchCV(keras_reg, param_grid, cv=5, random_state=42)
# rnd_search_cv.fit(X_train_scaled, Y_train_scaled, epochs=100, validation_data=(X_valid_scaled, Y_valid_scaled),
#                   callbacks=[keras.callbacks.EarlyStopping(patience=30)])

# print(rnd_search_cv.best_params_)

# model = rnd_search_cv.best_estimator_.model

model = keras.models.Sequential([
    layers.Dense(112, input_shape=X_train_scaled.shape[1:], activation='elu', kernel_initializer='normal'),
    layers.Dense(56, activation='elu', kernel_initializer='normal'),
    layers.Dense(1, kernel_initializer='normal')
])

model.summary()

model.compile(loss='mse', optimizer='sgd')
history = model.fit(X_train_scaled, Y_train_scaled, epochs=100, validation_data=(X_valid_scaled, Y_valid_scaled),
                    callbacks=[keras.callbacks.EarlyStopping(patience=30)])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.grid(True)
plt.show()

# =================== Accuracy and Evaluation =====================

# train set performance
Y_train_pred_scaled = model.predict(X_train_scaled)
Y_train_pred = scalerY.inverse_transform(Y_train_pred_scaled)

results = X_train.copy()
results["predicted"] = Y_train_pred
results["actual"] = Y_train
results = results[['predicted', 'actual']]
results['predicted'] = results['predicted'].round(0)
print(results)

# reset the index of DataFrame and use the default indexing (0 1 2 3...N-1)
results = pd.DataFrame.reset_index(results, drop=True)

# visualize predicted vs actual in train set
plt.plot(results['predicted'].head(100), label='predicted')
plt.plot(results['actual'].head(100), label='actual')
plt.xlabel('index in train set')
plt.ylabel('price')
plt.title('Predicted vs Actual in Train set')
plt.legend()
plt.show()

# test set performance
Y_test_pred_scaled = model.predict(X_test_scaled)
Y_test_pred = scalerY.inverse_transform(Y_test_pred_scaled)

# copy the DataFrame indexes
test_results = X_test.copy()
test_results["predicted"] = Y_test_pred
test_results["actual"] = Y_test
test_results = test_results[['predicted', 'actual']]
test_results['predicted'] = test_results['predicted'].round(0)
print(test_results)

# reset the index of DataFrame and use the default indexing (0 1 2 3...N-1)
test_results = pd.DataFrame.reset_index(test_results, drop=True)

# visualize predicted vs actual in test set
plt.plot(test_results['predicted'].head(100), label='predicted')
plt.plot(test_results['actual'].head(100), label='actual')
plt.xlabel('index in test set')
plt.ylabel('price')
plt.title('Predicted vs Actual in test set')
plt.legend()
plt.show()

# Error Metric summary
model_performance = pd.DataFrame(columns=['Train MAE', 'Train RMSE', 'Test MAE', 'Test RMSE'])

Train_MAE = mean_absolute_error(Y_train, Y_train_pred).round(0)
Train_RMSE = np.sqrt(mean_squared_error(Y_train, Y_train_pred)).round(0)

Test_MAE = mean_absolute_error(Y_test, Y_test_pred).round(0)
Test_RMSE = np.sqrt(mean_squared_error(Y_test, Y_test_pred)).round(0)

model_performance = model_performance.append({'Train MAE': Train_MAE,
                                              'Train RMSE': Train_RMSE,
                                              'Test MAE': Test_MAE,
                                              'Test RMSE': Test_RMSE},
                                             ignore_index=True)

print(model_performance)
