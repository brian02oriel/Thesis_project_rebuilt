import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
import joblib

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, roc_curve
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

def measuringPerformance(name, model, X, y, y_true, y_pred):    
    print("Measuring performance of: ", name)
    print("\n")
    
    # Cross validation score
    print("Cross validation score: ")
    print(cross_val_score(model, X, y, scoring="accuracy"))
    print('\n')
    
    # Confusion matrix
    y_train_predict = cross_val_predict(model, X, y, cv=5)
    print("Confusion matrix: ")
    print(confusion_matrix(y, y_train_predict))
    print('\n')
    
    # Precision and Recall
    print("Precision: ")
    print(precision_score(y, y_train_predict, average='weighted'))
    print("Recall: ")
    print(recall_score(y, y_train_predict, average='weighted'))
    print('\n')
    
    # Accuracy score
    print("Accuracy: ")
    print(accuracy_score(y_true, y_pred))
    print('\n')

    # F1 score
    print("F1 score: ")
    print(f1_score(y, y_train_predict, average='weighted'))


df_features = pd.read_csv('./images_features.csv', header=None)
df_paths = pd.read_csv('../Color Samples Dataset/dataset_register.csv')
df = pd.concat([df_features, df_paths], axis = 1)
df = df.sample(frac = 1)
train_n = math.floor(df.shape[0] * 80 / 100)
train_X = df.iloc[:train_n, ]
test_X = df.iloc[train_n:, ]
train_X_features = train_X.iloc[:, 0:2048].to_numpy()
test_X_features = test_X.iloc[:, 0:2048].to_numpy()
train_Y = df['class'][:train_n]
test_Y = df['class'][train_n:]
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(train_X_features, train_Y)
prediction = knn.predict(test_X_features)
measuringPerformance("KNN", knn, train_X_features, train_Y, test_Y, prediction)
joblib.dump(knn, 'knn_classifier.joblib')
print('Success...')
