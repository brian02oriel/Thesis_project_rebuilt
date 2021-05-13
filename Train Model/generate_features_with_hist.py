import cv2
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

def get_features(src):
    bgr_planes = cv2.split(src)
    histSize = 256
    histRange = (0, 256)
    accumulate = False
    b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
    return r_hist, g_hist, b_hist
    
def get_images(paths):
    images = []
    for path in paths:
        image = cv2.imread(path)
        if(image is None):
            print('Could not open or find the image')
            exit(0)
        images.append(image)
    return images

df = pd.read_csv('../Color Samples Dataset/dataset_register.csv')
paths = df['paths']
images = get_images(paths)
r_features = []
g_features = []
b_features = []

for image in images:
    r, g, b = get_features(image)
    r_features.append(r)
    g_features.append(g)
    b_features.append(b)
r_features = np.array(r_features)
r_features = np.squeeze(r_features)
g_features = np.array(g_features)
g_features = np.squeeze(g_features)
b_features = np.array(b_features)
b_features = np.squeeze(b_features)
np.savetxt('./r_features.csv', r_features, delimiter=",", header='')
np.savetxt('./g_features.csv', g_features, delimiter=",", header='')
np.savetxt('./b_features.csv', b_features, delimiter=",", header='')
print('Success...')