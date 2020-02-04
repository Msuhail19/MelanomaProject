# Import all libraries
import datetime
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

# This code initialises base_model as the inception v3 model without output layer
# We will create out own specific output layer
print(datetime.datetime.now())

# Model used is inception with imagenet weights by default
# Remove output layer as we are replacing it with out own
base_model = InceptionV3(weights='imagenet', include_top=False)

# Initialise variable classes
CLASSES = 2

# set pooling activation etc.
# Training new model
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.4)(x)

# Create output layer and add output layer to model.
predictions = Dense(CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Transfer learning, have all weights be none trainable except those connected to output nodes
# We aren't changing any layers that handle convolution for the moment
for layer in base_model.layers:
    layer.trainable = False

# Compile model with loss function and optimizers.
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define used variables.
EPOCHS = 10
BATCH_SIZE = 64
STEPS_PER_EPOCH = 20
VALIDATION_STEPS = 128
WIDTH = 299
HEIGHT = 299

# Define directories
TRAIN_DIR = 'E:/SkinDirectory/train'
TEST_DIR = 'E:/SkinDirectory/test'

# data prep
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

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

# File name is
MODEL_FILE = 'Inception.model'

# Begin fitting model
history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS)

model.save(MODEL_FILE)

print('Finished At : ')
print(datetime.datetime.now())
