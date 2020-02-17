# Import all libraries
import datetime
import keras as k
import tensorflow as tf
from IPython.core.display import display
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


# File name is
MODEL_FILE = 'InceptionV3' + '-RandomlyGen' + '-.model'

base_model = InceptionV3(include_top=False)

base_model.save(MODEL_FILE)
