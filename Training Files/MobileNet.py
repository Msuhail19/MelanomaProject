# Import all libraries
import datetime
import keras as k
import tensorflow as tf
from IPython.core.display import display
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

# This code initialises base_model as the inception v3 model without output layer
# We will create out own specific output layer
print(datetime.datetime.now())

# Initialise variable classes
CLASSES = 2

# Number of images in training and validation
TRAIN_COUNT = 9909
TEST_COUNT = 2636

# Define used variables.
EPOCHS = 20
BATCH_SIZE = 32
STEPS_PER_EPOCH = TRAIN_COUNT // BATCH_SIZE
VALIDATION_STEPS = TEST_COUNT // BATCH_SIZE
WIDTH = 224
HEIGHT = 224

# Define directories
TRAIN_DIR = 'E:\Attempt 6\Original'
TEST_DIR = 'E:\Attempt 6\Test'
filepath_epoch = "E:/MobileNetRand-BATCH " + str(BATCH_SIZE) + "-{epoch:02d}-.model"

# Model used is inception with imagenet weights by default
# Remove output layer as we are replacing it with out own
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=[HEIGHT, WIDTH, 3])

print(len(base_model.layers))

for layer in base_model.layers:
    layer.trainable = False
    # 60 layers works. more does not.
for layer in base_model.layers[:-50]:
    layer.trainable = True

print(len(base_model.trainable_weights))

# set pooling activation etc.
# Training new model
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.35)(x)

# Create output layer and add output layer to model.
predictions = Dense(CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


# Compile model with optimizer
rms = k.optimizers.RMSprop(learning_rate=0.01, decay=0.9, epsilon=0.1)
sgd = k.optimizers.SGD(momentum=0.9, learning_rate=0.005)
adam = k.optimizers.Adam(learning_rate=0.01, amsgrad=True)
model.compile(optimizer=rms,
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
    shuffle=True,
    class_mode='categorical')

#
validation_generator = validation_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical')

# Set Checkpoint
checkpoint = k.callbacks.callbacks.ModelCheckpoint(filepath_epoch, monitor='val_acc', verbose=1, save_best_only=False,
                                                   mode='max')
early_stopping = k.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=15, verbose=1)
reduce_lr = k.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, verbose=1, mode='min',
                                                    min_lr=0.00001)


# If an image is unreadable drop the batch
def my_gen(gen):
    while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except:
            pass


# Begin fitting model
class_weight = {0: 5.,
                1: 1.}

# Begin fitting model
history = model.fit_generator(
    my_gen(train_generator),
    epochs=EPOCHS,
    callbacks=[checkpoint],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=my_gen(validation_generator),
    validation_steps=VALIDATION_STEPS,
    class_weight=class_weight,
    verbose=1)

display(history.history)

# File name is
MODEL_FILE = 'MobileNet-' + str(BATCH_SIZE) + '-.model'

model.save(MODEL_FILE)

print('Finished At : ')
print(datetime.datetime.now())
