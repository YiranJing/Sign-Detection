"""
Model: Turn and Stop Sign Classification using NN
Author: Yiran Jing
Date: 2 Oct 2020
"""

# Example:
# python model_train.py


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


#
foler = "signed_map_test_2Oct"
# Predefined Hyper-parameter
learning_rate = 0.001
batch_size = 32
img_height = 240
img_width = 240
num_classes = 5
epochs = 7


def load_data(foler):
    """
    load train and test data from the given folder
    
    parameters:
        foler: folder path
        
    return:
        data_dir: relative path of train dataset
        test_data_dir: relative path of test dataset
    """
    print("Train data: \n")
    path = foler+"/train/stop"
    stop_image_count = len(listdir(path))
    print("There are {} colored stop turn images.".format(stop_image_count))
    path = foler+"/train/other"
    other_image_count = len(listdir(path))
    print("There are {} colored other turn images.".format(other_image_count))
    path = foler+"/train/left"
    left_image_count = len(listdir(path))
    print("There are {} colored left turn images.".format(left_image_count))
    path = foler+"/train/right"
    right_image_count = len(listdir(path))
    print("There are {} colored right turn images.".format(right_image_count))
    path = foler+"/train/park"
    park_image_count = len(listdir(path))
    print("There are {} colored park turn images.".format(park_image_count))
    data_dir = pathlib.Path(foler+"/train")
    #image_count = len(list(data_dir.glob('*/*.png')))
    image_count = len(list(data_dir.glob('*/*'))) # there are some png, also some jpg
    print("The total number of turning images are {} ".format(image_count))
    
    print("Test data: \n")
    path = foler+"/test/stop"
    stop_image_count = len(listdir(path))
    print("There are {} colored stop turn images.".format(stop_image_count))
    path = foler+"/test/other"
    other_image_count = len(listdir(path))
    print("There are {} colored other turn images.".format(other_image_count))
    path = foler+"/test/left"
    left_image_count = len(listdir(path))
    print("There are {} colored left turn images.".format(left_image_count))
    path = foler+"/test/right"
    right_image_count = len(listdir(path))
    print("There are {} colored right turn images.".format(right_image_count))
    path = foler+"/test/park"
    park_image_count = len(listdir(path))
    print("There are {} colored park turn images.".format(park_image_count))
    test_data_dir = pathlib.Path(foler+"/test")
    #image_count = len(list(data_dir.glob('*/*.png')))
    image_count = len(list(test_data_dir.glob('*/*'))) # there are some png, also some jpg
    print("The total number of test images are {} ".format(image_count))
    
    return data_dir, test_data_dir


def separate_train_val_test_data(data_dir, test_data_dir):
    """
    separate train, validation, and test data, and resize the image
    we split 20% data from train as the validation set

    
    parameters:
        data_dir: relative path of train dataset
        test_data_dir: relative path of test dataset
    
    return:
        train_ds: keras preprocessing image_dataset
        val_ds: keras preprocessing image_dataset
        test_ds: keras preprocessing image_dataset
    """

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2, # 20% as validation data
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        image_size=(img_height, img_width),
        seed=123,
        batch_size=batch_size)
        
    return train_ds, val_ds, test_ds


def visualization_train_data(class_names, train_ds):
    """
    Visualization training data
    
    parameters:
        class_names: list of string
        train_ds: c
    """
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")



def configure_dataset(train_ds, val_ds, test_ds):
    """
    Configure the dataset for performance
    
    parameters:
        train_ds: keras preprocessing image_dataset
        val_ds: keras preprocessing image_dataset
        test_ds: keras preprocessing image_dataset
    
    return:
        train_ds: keras preprocessing image_dataset
        val_ds: keras preprocessing image_dataset
        test_ds: keras preprocessing image_dataset
    """
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds
    


def train_CNN(train_ds, val_ds):
    """
    train CNN model
    
    parameters:
        train_ds: keras preprocessing image_dataset
        val_ds: keras preprocessing image_dataset
        
    return:
        cnn_1: model object
        cnn_history: model fit result
    """
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2) ## add EarlyStopping

    cnn_1 = Sequential([
              #data_augmentation,
              layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
              layers.Conv2D(16, 3, padding='same', activation='relu'),
              layers.MaxPooling2D(),
              layers.Conv2D(32, 3, padding='same', activation='relu'),
              layers.MaxPooling2D(),
              layers.Conv2D(64, 3, padding='same', activation='relu'),
              layers.MaxPooling2D(),
              #layers.Dropout(drop_out), # drop out rate
              layers.Flatten(),
              layers.Dense(128, activation='relu'),
              layers.Dense(num_classes)
                ])

    cnn_1.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    cnn_history = cnn_1.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs,
      callbacks=[callback] # EarlyStopping
    )
    
    
    return cnn_1, cnn_history




def plot_train_result(model_result, epoch):
    """
    plot the Training and Validation Accuracy and Training and Validation Loss
    
    parameters:
        model_result: model fit result
        epoch: integer
    """
    acc = model_result.history['accuracy']
    val_acc = model_result.history['val_accuracy']

    loss=model_result.history['loss']
    val_loss=model_result.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def plot_normalized_confusion_matrix(y_true, y_pred, title, class_names):
    """
    helper function of draw_missclassification_images
    draw confusion matrix based on the raw matrix, adding label
    
    parameters:
        class_names: list of string
        title: string
        y_true: panda series
        y_pred: panda series
    """
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
    """
    Plot confusion matrix and classification report
    
    parameters:
        class_names: list of string
        val_ds: keras preprocessing image_dataset
    """
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



def draw_missclassification_images(class_name, model, test):
    """
    Plot confusion matrix and classification report
    
    parameters:
        class_names: list of string
        model: model object
        test: keras preprocessing image_dataset
    """

    # test dataset 
    directory = test_data_dir.glob(class_name+'/*.jpg')
    image_list = list(test_data_dir.glob(class_name+'/*.jpg'))
    #test = keras.preprocessing.image_dataset_from_directory(
    #        test_data_dir, image_size=(img_height, img_width),seed=123
    #)
    test = test_ds


    count = 0
    for image in directory:
    
        picture = Image.open(image)
    
        img = keras.preprocessing.image.load_img(
            image, target_size=(img_height, img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        if class_names[np.argmax(score)] != class_name: # miss classification
            print("Image:", str(image).split('/')[-1])
            print("True class: ", class_name)
            print("Predicted class:", class_names[np.argmax(score)], ", with Prediction CI", round(100 * np.max(score), 2))

            # display image
            pil_im = Image.open(str(image_list[count]), 'r')
            display(pil_im)
            count +=1

            print("\n\n")



if __name__ == "__main__":
    
    # Load dataset
    data_dir, test_data_dir = load_data(foler)
    train_ds, val_ds, test_ds = separate_train_val_test_data(data_dir, test_data_dir)
    ## class name
    class_names = train_ds.class_names
    print(class_names)
    
    ## visualization some train images
    visualization_train_data(class_names, train_ds)
    
    # Feature Engineering
    # Configure the dataset for performance
    train_ds, val_ds, test_ds = configure_dataset(train_ds, val_ds, test_ds)
    
    ## Train Model
    cnn_1, cnn_history = train_CNN(train_ds, val_ds)
    
    ## Visualize results
    plot_train_result(cnn_history, epochs)
    
    ## Model evaluation
    draw_confusion_matrix(cnn_1, test_ds)
    # save model
    cnn_1.save('sign_2Oct_nofeatureEngineer.h5')

    ## Check misclassification images in the test data
    ## Misclassification Stop
    #draw_missclassification_images('stop', cnn_1, test_ds)
    ## Misclassification Left-turning
    #draw_missclassification_images('left', cnn_1, test_ds)
    ## Misclassification right-turning
    #draw_missclassification_images('right', cnn_1, test_ds)
    ## Misclassification park
    #draw_missclassification_images('park', cnn_1, test_ds)
    ## Misclassification other
    #draw_missclassification_images('other', cnn_1, test_ds)


    
    



