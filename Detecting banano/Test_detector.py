import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
from LocalBinaryPattern import LocalBinaryPatterns
from joblib import load

def creating_dataset(data_path, label, X, y):
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    
    # Preparing the radius and the no. of points of the LBP descriptor
    radius = 3
    no_points = 8 * radius

    desc = LocalBinaryPatterns(no_points, radius)

    for files in onlyfiles:
        image_path = data_path + files
        images = cv2.imread(image_path, 0)
        images = cv2.resize(images, (150,150))
        hist = desc.describe(images)
        X.append(hist)
        y.append(label)
    return X, y

X, y = [], []
X, y = creating_dataset('../Dataset/Green banana/', 1, X, y)
X, y = creating_dataset('../Dataset/Green segment/', 1, X, y)
X, y = creating_dataset('../Dataset/Market segment/', 1, X, y)
X, y = creating_dataset('../Dataset/Negative reference/', 0, X, y)

knn = load('./Banana_bin_classifier.joblib')
prediction = knn.predict(X)
print(prediction)