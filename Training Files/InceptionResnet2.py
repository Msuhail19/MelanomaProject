# Import all libraries
import datetime
import keras as k
from IPython.core.display import display
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.callbacks import ModelCheckpoint, CSVLogger
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
EPOCHS = 8
BATCH_SIZE = 16
STEPS_PER_EPOCH = TRAIN_COUNT//BATCH_SIZE
VALIDATION_STEPS = TEST_COUNT//BATCH_SIZE
WIDTH = 299
HEIGHT = 299
LR = 0.02

# Define directories
TRAIN_DIR = 'E:\Attempt 6\Original'
TEST_DIR = 'E:\Attempt 6\Test'

# Model used is inception with imagenet weights by default
# Remove output layer as we are replacing it with out own
base_model = InceptionResNetV2(include_top=False,
                            weights='imagenet',
                            input_shape=(299,299,3))

print(len(base_model.layers))
print(len(base_model.trainable_weights))

# Unfreeze the last three inception modules
for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers[:-200]:
    layer.trainable = True

print(len(base_model.trainable_weights))

# set pooling activation etc.
# Training new model
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.5)(x)

# Create output layer and add output layer to model.
predictions = Dense(CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

rms = k.optimizers.RMSprop(learning_rate=LR, decay=0.9, epsilon=0.1)
model.compile(optimizer=rms,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# data prep
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=360,
    brightness_range=[0.7, 1.3],
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    )

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical',)

#
validation_generator = validation_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical')


# Set Checkpoint
filepath = "E:\InceptionResNet2-BATCH " + str(BATCH_SIZE) + "-{epoch:02d}-.model"
checkpoint = k.callbacks.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False,
                                                   mode='max')


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

csv_logger = CSVLogger('InceptionResnetV2-History-.csv', append=True)

history = model.fit_generator(
    my_gen(train_generator),
    epochs=EPOCHS,
    callbacks=[csv_logger, checkpoint],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=my_gen(validation_generator),
    validation_steps=VALIDATION_STEPS,
    class_weight=class_weight,
    shuffle=True,
    verbose=1)

display(history.history)

# File name is
MODEL_FILE = 'FinalResNetInception.model'

model.save(MODEL_FILE)

print('Finished At : ')
print(datetime.datetime.now())
