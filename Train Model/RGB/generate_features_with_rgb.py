import cv2
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

def get_features_image(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    src = src.reshape((src.shape[0] * src.shape[1], 3))
    return src[0], src[1], src[2]

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

r_features = []
g_features = []
b_features = []

for image in images:
    r, g, b = get_features_image(image)
    r_features.append(r)
    g_features.append(g)
    b_features.append(b)

r_features = np.array(r_features)
r_features = np.squeeze(r_features)
g_features = np.array(g_features)
g_features = np.squeeze(g_features)
b_features = np.array(b_features)
b_features = np.squeeze(b_features)
np.savetxt('./image_r_features.csv', r_features, delimiter=",", header='')
np.savetxt('./image_g_features.csv', g_features, delimiter=",", header='')
np.savetxt('./image_b_features.csv', b_features, delimiter=",", header='')
print('Success...')