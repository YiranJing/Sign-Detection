"""
Test model performance
"""

import matplotlib.pyplot as plt
from os import listdir, rename, listdir
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


## load model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filename = "sign_2Oct_nofeatureEngineer.h5" 
# load model
cnn_1 = load_model(filename)
class_names = ['left', 'other', 'park', 'right', 'stop']
# summarize model.
#cnn_1.summary()

## load test data
foler = "test_data"

print("Test data: \n")
path = foler+"/stop"
stop_image_count = len(listdir(path))
print("There are {} colored stop images.".format(stop_image_count))
path = foler+"/other"
other_image_count = len(listdir(path))
print("There are {} colored other images.".format(other_image_count))
path = foler+"/left"
left_image_count = len(listdir(path))
print("There are {} colored left turn images.".format(left_image_count))
path = foler+"/right"
right_image_count = len(listdir(path))
print("There are {} colored right turn images.".format(right_image_count))
path = foler+"/park"
park_image_count = len(listdir(path))
print("There are {} colored parkimages.".format(park_image_count))

path = foler+"/stop"
test_data_dir = pathlib.Path(foler)
image_count = len(list(test_data_dir.glob('*/*'))) # there are some png, also some jpg
print("The total number of test images are {} ".format(image_count))


## Data pre-processing
# Predefined Hyper-parameter 
batch_size = 32
img_height = 240
img_width = 240

AUTOTUNE = tf.data.experimental.AUTOTUNE
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_data_dir, 
  image_size=(img_height, img_width),
  seed=123, 
  batch_size=batch_size)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

"""
Model evaluation
"""
def plot_normalized_confusion_matrix(y_true, y_pred, title, class_names):
    #y_true = np.concatenate([y for x, y in val_ds], axis=0)
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    target_names = class_names
    fig, ax = plt.subplots(figsize=(8,5))
    fig.suptitle(title,fontsize=20)
    ax.set_xlabel("Preidicted_Label")
    ax.set_ylabel('True_Label')
    plt.xticks(np.arange(0, 60), np.arange(1,60))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, 
            yticklabels=target_names, linewidths=.8,)
    #image_path = 'image/' + model +'_normalized_confusion_matrix.png'
    #fig.savefig(image_path)   
    #plt.close(fig)
    plt.show()  
    
def draw_confusion_matrix(model, val_ds):
    #Confution Matrix and Classification Report
    Y_pred = model.predict(val_ds)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix, x label is Predicted_Label, y is True_Label')
    y_true = np.concatenate([y for x, y in val_ds], axis=0)
    plot_normalized_confusion_matrix(y_true, y_pred, "Normalized Confusion Matrix", class_names) # self-defined function
    
    print('\n\nClassification Report bazed on validation data')
    #target_names = train_ds.class_names
    target_names = class_names
    print(classification_report(y_true, y_pred, target_names=target_names))

    
draw_confusion_matrix(cnn_1, test_ds)