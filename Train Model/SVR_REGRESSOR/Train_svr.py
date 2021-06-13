import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics

def measure_regressor_error(y_test, prediction):
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

def days_between(d1, d2):
    d1 = datetime.strptime(d1[:19], "%Y-%m-%d %H:%M:%S")
    d2 = datetime.strptime(d2[:19], "%Y-%m-%d %H:%M:%S")
    return abs((d2 - d1).days)

df_features = pd.read_csv('./image_hist_features.csv', header=None)
df_paths = pd.read_csv('../../Full Samples Dataset/dataset_register.csv')

days_difference = []
for index, row in df_paths.copy().iterrows():
  difference = days_between(row['datetime'], row['last_capture'])
  days_difference.append(difference)
df_paths['difference'] = days_difference
X=df_features
y=df_paths['difference']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

svr = SVR().fit(X_train, y_train)
prediction = svr.predict(X_test)
print(prediction)
measure_regressor_error(y_test, prediction)
joblib.dump(svr, 'svr_regressor.joblib')
print('success...')


