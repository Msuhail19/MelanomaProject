# Import all libraries
import datetime
import keras as k
from IPython.core.display import display
from keras.applications.inception_v3 import InceptionV3, preprocess_input
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
TRAIN_COUNT = 48160
TEST_COUNT = 12041

# Define used variables.
EPOCHS = 8
BATCH_SIZE = 32
STEPS_PER_EPOCH = TRAIN_COUNT // BATCH_SIZE
VALIDATION_STEPS = TEST_COUNT // BATCH_SIZE
WIDTH = 299
HEIGHT = 299

# Define directories
TRAIN_DIR = 'E:/Generate 2/Train'
TEST_DIR = 'E:/Generate 2/Test'
filepath_epoch = "Models Gen 2/Inception3-BATCH " + str(BATCH_SIZE) + "-{epoch:02d}-.model"

# Model used is inception with imagenet weights by default
# Remove output layer as we are replacing it with out own
base_model = InceptionV3(weights='imagenet', include_top=False)

print(len(base_model.layers))
print(len(base_model.trainable_weights))

# Unfreeze the last three inception modules
for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers[:-180]:
    layer.trainable = True

print(len(base_model.trainable_weights))

# set pooling activation etc.
# Training new model
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.4)(x)

# Create output layer and add output layer to model.
predictions = Dense(CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)



opt = k.optimizers.RMSprop(learning_rate=0.007, decay=0.9, epsilon=0.1)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# data prep
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    brightness_range=[0.6, 1.4],
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    vertical_flip=True,
    horizontal_flip=True,
    fill_mode='reflect')

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
checkpoint = k.callbacks.callbacks.ModelCheckpoint(filepath_epoch, monitor='val_acc', verbose=1, save_best_only=False,
                                                   mode='max')


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
bestcheckpoint = ModelCheckpoint('saved_model/InceptionBestVersion.model', verbose=1, save_best_only=True,
                                 monitor='val_acc', mode='max')

# Begin fitting model
history = model.fit_generator(
    my_gen(train_generator),
    epochs=EPOCHS,
    callbacks=[checkpoint],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=my_gen(validation_generator),
    validation_steps=VALIDATION_STEPS,
    verbose=1)

display(history.history)

# File name is
MODEL_FILE = 'Inception' + str(BATCH_SIZE) + '.model'

model.save(MODEL_FILE)

print('Finished At : ')
print(datetime.datetime.now())
