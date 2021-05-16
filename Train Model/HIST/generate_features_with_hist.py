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
    b_hp = len([1 for b in b_hist if b > 0]) * 100 / src_count
    g_hp = len([1 for g in g_hist if g > 0]) * 100 / src_count
    r_hp = len([1 for r in r_hist if r > 0]) * 100 / src_count
    return b_hp, g_hp, r_hp

def get_features_hsv(src):
    hsv = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    print(hist)
    return hist
    
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
hsv_features = []

for image in images:
    r, g, b = get_features_rgb(image)
    rgb_features.append([r, g, b])
rgb_features = np.array(rgb_features)
rgb_features = np.squeeze(rgb_features)
np.savetxt('./image_rgb_features.csv', rgb_features, delimiter=",", header='')



#for image in images:
#    features = get_features_hsv(image)
#    hsv_features.append(features)
#rgb_features = np.array(rgb_features)
#rgbfeatures = np.squeeze(rgb_features)
#np.savetxt('./image_rgb_features.csv', rgb_features, delimiter=",", header='')
print('Success...')