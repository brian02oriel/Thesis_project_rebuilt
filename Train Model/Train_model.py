import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.cluster import KMeans
import joblib

df_features = pd.read_csv('./images_features.csv', header=None)
df_paths = pd.read_csv('../Color Samples Dataset/dataset_register.csv')
df = pd.concat([df_features, df_paths], axis = 1)
df = df.sample(frac = 1)
train_n = math.floor(df.shape[0] * 80 / 100)
train_X = df.iloc[:train_n, ]
test_X = df.iloc[train_n:, ]
train_X_features = train_X.iloc[:, 0:2048].to_numpy()
test_X_features = test_X.iloc[:, 0:2048].to_numpy()

kmeans = KMeans(n_clusters=5)
kmeans.fit(train_X_features)
prediction = kmeans.predict(test_X_features)
centers = kmeans.cluster_centers_
class_assignment = kmeans.predict(df.iloc[:, 0:2048].to_numpy())
df_paths['class'] = class_assignment
df_paths.to_csv('./dataset_register_with_classes.csv')

joblib.dump(kmeans, 'kmeans_class_generator.joblib')
print('Success...')

