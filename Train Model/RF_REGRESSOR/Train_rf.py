import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

def measure_regressor_error(y_test, prediction):
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

def days_between(d1, d2):
    d1 = datetime.strptime(d1[:19], "%Y-%m-%d %H:%M:%S")
    d2 = datetime.strptime(d2[:19], "%Y-%m-%d %H:%M:%S")
    return abs((d2 - d1).days)

def display(parameters, rf, X_train,y_train):
    cv = GridSearchCV(rf,parameters,cv=5)
    cv.fit(X_train,y_train)
    print(f'Best parameters are: {cv.best_params_}')
    print("\n")
    return cv.best_params_


df_features = pd.read_csv('./image_hist_features.csv', header=None)
df_paths = pd.read_csv('../../Color Samples Dataset/dataset_register.csv')

days_difference = []
for index, row in df_paths.copy().iterrows():
  difference = days_between(row['datetime'], row['last_capture'])
  days_difference.append(difference)
df_paths['difference'] = days_difference
X=df_features
y=df_paths['difference']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

parameters = {
    "n_estimators":[5,10,50,100,250],
    "max_depth":[2,4,8,16,32,None],
}

rf = RandomForestRegressor()
best_params = display(parameters, rf, X_train,y_train)
rf = RandomForestRegressor(max_depth=best_params.get('max_depth'), n_estimators=best_params.get('n_estimators'))
rf.fit(X_train, y_train)
prediction = rf.predict(X_test)
print(prediction)
measure_regressor_error(y_test, prediction)
joblib.dump(rf, 'rf_regressor.joblib')
print('success...')


