import cv2
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

def get_features_rgb(src, i):
    src_row1 = cv2.hconcat([src, src, src, src])
    src_row2 = cv2.hconcat([src, src, src, src])
    src_row3 = cv2.hconcat([src, src, src, src])
    src_row4 = cv2.hconcat([src, src, src, src])
    src = cv2.vconcat([src_row1, src_row2, src_row3, src_row4])
    cv2.imwrite('./Frame_generated/' + i + '.jpg', src)
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
        image = cv2.resize(image, (150, 150))
        if(image is None):
            print('Could not open or find the image')
            exit(0)
        images.append(image)
    return images

df = pd.read_csv('../../Full Samples Without Rotation/dataset_register.csv')
paths = df['paths']
images = get_images(paths)
bgr_features = []

for i, image in enumerate(images):
    b, g, r = get_features_rgb(image, str(i))
    bgr_feature = b + g + r
    bgr_features.append(bgr_feature)
bgr_features = np.array(bgr_features)
bgr_features = np.squeeze(bgr_features)
np.savetxt('./image_hist_features.csv', bgr_features, delimiter=",", header='')
print('Success...')