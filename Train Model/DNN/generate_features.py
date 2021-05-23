import cv2
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

def get_features(img, net):
  layers = net.getLayerNames() 
  blob = cv2.dnn.blobFromImage(np.asarray(img), 1, (224, 224), (104, 117, 123))
  net.setInput(blob)
  preds = net.forward(outputName='pool5')
  return preds[0]

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
model_file = './ResNet-50-model.caffemodel'
deploy_prototxt = './ResNet-50-deploy.prototxt'
net = cv2.dnn.readNetFromCaffe(deploy_prototxt, model_file)
images_features = []
for image in images:
    features = get_features(image, net)
    images_features.append(features)
np_features = np.array(images_features)
np_features = np.squeeze(np_features)
np.savetxt('./images_features_with_dnn.csv', np_features, delimiter=",", header='')
print('Success...')