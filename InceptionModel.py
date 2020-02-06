# Import all libraries
import datetime
import keras as k
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

# Unfreeze the last three inception modules
for layer in model.layers[:229]:
    layer.trainable = False
for layer in model.layers[229:]:
    layer.trainable = True

# Compile model with loss function and optimizers.
opt2 = k.optimizers.SGD()
opt = k.optimizers.RMSprop(learning_rate=0.003, decay=0.9, epsilon=0.1)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define used variables.
EPOCHS = 10
BATCH_SIZE = 32
STEPS_PER_EPOCH = 1406
VALIDATION_STEPS = 32
WIDTH = 299
HEIGHT = 299

# Define directories
TRAIN_DIR = 'E:/Generate/Train'
TEST_DIR = 'E:/Generate/Test'

# data prep
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    brightness_range=[0.5,1.5],
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='reflect')

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    brightness_range=[0.5, 1.5],
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='reflect')

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

# Set Checkpoint
filepath = "Inception-BATCH " + str(BATCH_SIZE) + "-0515BRIGHTNESS-{epoch:02d}-.model"
checkpoint = k.callbacks.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')

def my_gen(gen):
    while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except:
            pass




# Begin fitting model
history = model.fit_generator(
    my_gen(train_generator),
    epochs=EPOCHS,
    callbacks=[checkpoint],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=my_gen(validation_generator),
    validation_steps=VALIDATION_STEPS)

model.save(MODEL_FILE)

print('Finished At : ')
print(datetime.datetime.now())
