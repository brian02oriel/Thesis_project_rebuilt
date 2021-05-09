import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd

def creating_samples_dataset(data_path, target_path, df):
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    for files in onlyfiles:
        image_path = data_path + files
        image = cv2.imread(image_path)
        image = cv2.resize(image, (350,350))
        image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
        w = 100
        h = 100
        center_x = image.shape[0] / 2
        center_y = image.shape[1] / 2
        x = (center_x - 50) - w/2
        y = center_y - h/2
        crop_img = image[int(y):int(y+h), int(x):int(x+w)]
        cv2.imwrite(target_path + files, crop_img)
        file_name = files.split('.')
        id = file_name[0]
        df.at[int(id), 'paths'] = target_path + files
    return df

green_banana_df = pd.read_csv('../Environmental captures/Green_banana.csv')
green_banana_df['paths'] = ''

green_segment_df = pd.read_csv('../Environmental captures/Green_segment.csv')
green_segment_df['paths'] = ''

market_segment_df = pd.read_csv('../Environmental captures/Market_segment.csv')
market_segment_df['paths'] = ''


green_banana_df = creating_samples_dataset('../Dataset/Green banana/', '../Color Samples Dataset/Green banana/', green_banana_df.copy())
green_segment_df = creating_samples_dataset('../Dataset/Green segment/', '../Color Samples Dataset/Green segment/', green_segment_df.copy())
market_segment_df = creating_samples_dataset('../Dataset/Market segment/', '../Color Samples Dataset/Market segment/', market_segment_df.copy())
df = pd.concat([green_banana_df, green_segment_df,market_segment_df])
print(df)
df.to_csv('../Color Samples Dataset/dataset_register.csv')
print('success')        