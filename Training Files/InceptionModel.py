# Import all libraries
import datetime
import keras as k
import numpy
import pandas
import sklearn
from IPython.core.display import display
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator



# Initialise variable classes
CLASSES = 2

# Number of images in training and validation
TRAIN_COUNT = 9909
TEST_COUNT = 2636

# Define used variables.
EPOCHS = 15
BATCH_SIZE = 32
STEPS_PER_EPOCH = TRAIN_COUNT // BATCH_SIZE
VALIDATION_STEPS = TEST_COUNT // BATCH_SIZE
WIDTH = 299
HEIGHT = 299
LEARNING_RATE = 0.01

# Define directories
TRAIN_DIR = 'E:\Attempt 6\Original'
TEST_DIR = 'E:\Attempt 6\Test'
filepath_epoch = R"E:\Attempt 6/Inception3-70-30- " + str(BATCH_SIZE) + "-{epoch:02d}-.model"

# Model used is inception with imagenet weights by default
# Remove output layer as we are replacing it with out own
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=[HEIGHT,WIDTH,3])

print(len(base_model.layers))
print(len(base_model.trainable_weights))

# Unfreeze inception modules
for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers[:-15]:
    layer.trainable = True


print(len(base_model.trainable_weights))
print(LEARNING_RATE)

# set pooling activation etc.
# Training new model
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.5)(x)


# Create output layer and add output layer to model.
predictions = Dense(CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


opt = k.optimizers.RMSprop(learning_rate=LEARNING_RATE, decay=0.9, epsilon=0.1)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# data prep
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    brightness_range=[0.7, 1.3],
    zoom_range=0.1,
    rotation_range=360,
    vertical_flip=True,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

#
validation_generator = validation_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Set Checkpoint
checkpoint = k.callbacks.callbacks.ModelCheckpoint(filepath_epoch, monitor='val_accuracy', verbose=1, save_best_only=False,
                                                   mode='max')

from datetime import datetime

now = datetime.now()
date_time = now.strftime("%m-%d-%Y")

csv_logger = CSVLogger('InceptionV3-15-30- ' + date_time + '.csv', append=True)


# If an image is unreadable drop the batch
def my_gen(gen):
    while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except:
            pass


# Have a checkpoint to store the best
# Save the model with best weights
bestcheckpoint = ModelCheckpoint('E:/Models/InceptionBestVersion.h5', verbose=1, save_best_only=True,
                                 monitor='val_loss', mode='min')

# Begin fitting model
class_weight = {0: 5.,
                1: 1.}

history = model.fit_generator(
    my_gen(train_generator),
    epochs=EPOCHS,
    callbacks=[csv_logger, checkpoint],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=my_gen(validation_generator),
    shuffle=True,
    validation_steps=VALIDATION_STEPS,
    class_weight=class_weight,
    verbose=1)

display(history.history)
SAVE = 'Newest history' + '.csv'
pandas.DataFrame.from_dict(history.history).to_csv(SAVE, index=False)


# File name is
MODEL_FILE = 'Inception' + str(BATCH_SIZE) + '.model'
model.save(MODEL_FILE)

import matplotlib.pyplot as plt

def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title("Training and validation accuracy")

    plt.figure()
    plt.plot(epochs, loss, 'b')
    plt.plot(epochs, val_loss, 'r')
    plt.title("Training and validation loss")

    plt.show()
    plt.savefig("Training and validation accuracy.png")


plot_training(history)

print('Finished At : ')
print(datetime.now())