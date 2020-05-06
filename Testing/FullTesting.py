import tensorflow as tf
from keras.applications import InceptionV3

tf.config.experimental.set_visible_devices([], 'GPU')

import numpy as np
import keras
from keras import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from keras.models import load_model
from glob import glob

# This script is to test the models created on all sets of images.

# Dimensions of image
HEIGHT = 224
WIDTH = 224

# Model directory
MODEL_FILE = R'E:/MobileNetRand-BATCH 32-01-.model'

def getInceptionModel():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=[HEIGHT, WIDTH, 3])
    # set pooling activation etc.
    # Training new model
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.3)(x)

    # Create output layer and add output layer to model.
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def getMobileNetModel():
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# Load model
loaded_model = getMobileNetModel()
loaded_model.load_weights(MODEL_FILE)


# Directories of images to test against
benign_imgs = glob(R'E:\Attempt 6\Original\Naevus/*')
benign_imgs2 = glob('E:/Attempt 6/Test/Naevus/*')
mal_imgs = glob(R'E:\Attempt 6\Original\Melanoma/*')
mal_imgs2 = glob('E:/Attempt 6/Test/Melanoma/*')

# Method to predict and return predictions
def predict(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = loaded_model.predict(x)
    return preds[0]

# Method to take a list of images return no of benign and malignant
def iterate(list):
    print(len(list))
    benign_no = 0
    mal_no = 0
    count = 0
    for item in list:
        img = image.load_img(item, target_size=(HEIGHT, WIDTH))
        preds = predict(img)
        keras.backend.clear_session()
        count = count + 1
        print(str(count) + ' ' + str(preds[0]) + ' ' + str(preds[1]))
        print(item)
        if preds[1] > preds[0]:
            print('Benign')
            benign_no = benign_no + 1
        else:
            print('Malignant')
            mal_no = mal_no + 1

    return benign_no, mal_no

# For each image dir get accuracy,
#
# print('Predicting Malignant Images ... ')
# ben_no, mal_no = iterate(mal_imgs)
# acc3 = mal_no/len(mal_imgs)
# print('Accuracy is ' + str(acc3))

print('Predicting Malignant Images 2 ... ')
ben_no, mal_no = iterate(mal_imgs2)
acc4 = mal_no/len(mal_imgs2)
print('Accuracy is ' + str(acc4))

# print('Predicting Benign ...')
# ben_no , mal_no = iterate(benign_imgs)
# acc1 = ben_no/len(benign_imgs)
# print('Accuracy is ' + str(acc1))

print('Predicting Benign 2 ...')
ben_no, mal_no = iterate(benign_imgs2)
acc2 = ben_no/len(benign_imgs2)
print('Accuracy is ' + str(acc2))


print('Malignant acc in order is : ' + str() + ' , ' + str(acc4))
print('Benign acc in order is : ' + str() + ',' + str(acc2))



