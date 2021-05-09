import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from LocalBinaryPattern import LocalBinaryPatterns
from joblib import dump
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, roc_curve
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

def plot_roc_curve(fpr, tpr, label=None):
    plt.title("ROC Curve")
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
def measuringPerformance(name, model, X, y, y_true, y_pred):    
    print("Measuring performance of: ", name)
    print("\n")
    
    # Cross validation score
    print("Cross validation score: ")
    print(cross_val_score(model, X, y, scoring="accuracy"))
    print('\n')
    
    # Confusion matrix
    y_train_predict = cross_val_predict(model, X, y, cv=3)
    print("Confusion matrix: ")
    print(confusion_matrix(y, y_train_predict))
    print('\n')
    
    # Precision and Recall
    print("Precision: ")
    print(precision_score(y, y_train_predict))
    print("Recall: ")
    print(recall_score(y, y_train_predict))
    print('\n')
    
    # Accuracy score
    print("Accuracy: ")
    print(accuracy_score(y_true, y_pred))
    print('\n')

    # F1 score
    print("F1 score: ")
    print(f1_score(y, y_train_predict))
    
    # ROC curve
    y_probs_rf = cross_val_predict(model, X, y, cv=3, method="predict_proba")
    y_scores_rf = y_probs_rf[:, 1]
    fpr_rf, tpr_rf, thresh_rf = roc_curve(y, y_scores_rf)
    plot_roc_curve(fpr_rf, tpr_rf)
    plt.show()
    
    # ROC AUC score
    print("ROC AUC score: ")
    print(roc_auc_score(y, y_scores_rf))

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(len(X_train), len(X_test), len(y_train), len(y_test))

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
clfp = clf.predict(X_test)
print(clfp)
measuringPerformance("KNN", clf, X_train, y_train, y_test, clfp)

dump(clf, 'Banana_bin_classifier.joblib')
print("Model successfully saved")
