import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.cluster import KMeans
import joblib
import math

def calculate_wcss(data):
    wcss = []
    for n in range(2, 21):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)
    return wcss

def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]
    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    return distances.index(max(distances)) + 2


rgb_features = pd.read_csv('./image_hist_features.csv', header=None)

df_paths = pd.read_csv('../../Color Samples Dataset/dataset_register.csv')
df = pd.concat([rgb_features, df_paths], axis = 1)
print(df)
df = df.sample(frac = 1)
train_n = math.floor(df.shape[0] * 80 / 100)
train_X = df.iloc[:train_n, ]
test_X = df.iloc[train_n:, ]
train_X_features = train_X.iloc[:, 0:768].to_numpy()
test_X_features = test_X.iloc[:, 0:768].to_numpy()

# calculating the within clusters sum-of-squares for 19 cluster amounts
sum_of_squares = calculate_wcss(train_X_features)

# calculating the optimal number of clusters
n = optimal_number_of_clusters(sum_of_squares)
print(n)

kmeans = KMeans(n_clusters=5)
kmeans.fit(train_X_features)
prediction = kmeans.predict(test_X_features)
centers = kmeans.cluster_centers_
class_assignment = kmeans.predict(df.iloc[:, 0:768].to_numpy())
df_paths['class'] = class_assignment
df_paths.to_csv('./dataset_register_with_classes_hist.csv')

joblib.dump(kmeans, 'kmeans_class_generator_hist.joblib')
print('Success...')