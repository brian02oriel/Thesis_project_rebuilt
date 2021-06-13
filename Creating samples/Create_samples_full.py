import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd

def get_image_center(image):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    return cX, cY, h, w

def augment_dataset(image):
    (cX, cY, h, w) = get_image_center(image)
    augmented_images = []
    rotation = 15
    for x in range(5):
        M = cv2.getRotationMatrix2D((cX, cY), rotation, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented_images.append(rotated)
        rotation = rotation + 15
    return augmented_images


def creating_samples_dataset(data_path, target_path, df):
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    for files in onlyfiles:
        image_path = data_path + files
        image = cv2.imread(image_path)
        image = cv2.resize(image, (350,350))
        image = cv2.medianBlur(image,5)
        #print('target: ', target_path + files)
        #cv2.imshow('im', image)
        cv2.imwrite(target_path + files, image)
        file_name = files.split('.')
        id = file_name[0]
        row_index = df.index[df['id'] == id][0]
        df.at[row_index, 'paths'] = '../' + target_path + files
        augmented_images = augment_dataset(image)
        rotation = 15
        for augmented_image in augmented_images:
            augmented_id = str(id) + '_'  + str(rotation)
            augmented_path = target_path + augmented_id + '.jpg'
            #cv2.imshow('augmented', augmented_image)
            #print('augmented: ', augmented_path)
            cv2.imwrite(augmented_path, augmented_image)
            row_index = df.index[df['id'] == augmented_id][0]
            df.at[row_index, 'paths'] = '../' + augmented_path
            rotation = rotation + 15
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

     

    return df

green_banana_df = pd.read_csv('../Environmental captures/Green_banana.csv')
green_banana_df['paths'] = ''
green_segment_df = pd.read_csv('../Environmental captures/Green_segment.csv')
green_segment_df['paths'] = ''
market_segment_df = pd.read_csv('../Environmental captures/Market_segment.csv')
market_segment_df['paths'] = ''
green_banana_df = creating_samples_dataset('../Dataset/Green banana/', '../Full Samples Dataset/Green banana/', green_banana_df.copy())
green_segment_df = creating_samples_dataset('../Dataset/Green segment/', '../Full Samples Dataset/Green segment/', green_segment_df.copy())
market_segment_df = creating_samples_dataset('../Dataset/Market segment/', '../Full Samples Dataset/Market segment/', market_segment_df.copy())
df = pd.concat([green_banana_df, green_segment_df,market_segment_df])
df.to_csv('../Full Samples Dataset/dataset_register.csv')
print('success')