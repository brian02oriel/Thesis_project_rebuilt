import cv2
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

def get_features_rgb(src):
    src_count = src.size
    histSize = 256
    histRange = (0, 256)
    accumulate = False
    b_hist = cv2.calcHist(src, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv2.calcHist(src, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv2.calcHist(src, [2], None, [histSize], histRange, accumulate=accumulate)
    b_hp = [b * 100 / src_count for b in b_hist]
    g_hp = [g * 100 / src_count for g in g_hist]
    r_hp = [r * 100 / src_count for r in r_hist]
    return b_hp, g_hp, r_hp

def get_images(paths):
    images = []
    for path in paths:
        image = cv2.imread(path)
        if(image is None):
            print('Could not open or find the image')
            exit(0)
        images.append(image)
    return images

df = pd.read_csv('../../Color Samples Dataset/dataset_register.csv')
paths = df['paths']
images = get_images(paths)
rgb_features = []

for image in images:
    r, g, b = get_features_rgb(image)
    rgb_feature = r + g + b
    rgb_features.append(rgb_feature)
rgb_features = np.array(rgb_features)
rgb_features = np.squeeze(rgb_features)
np.savetxt('./image_hist_features.csv', rgb_features, delimiter=",", header='')
print('Success...')