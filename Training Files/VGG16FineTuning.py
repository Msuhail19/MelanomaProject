# Import all libraries
import keras as k
import pandas
import tensorflow
from IPython.core.display import display
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, normalization, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa

from datetime import datetime

CLASSES = 2

# Number of images in training and validation
TRAIN_COUNT = 9909
TEST_COUNT = 2636

# Define used variables.
EPOCHS = 30
BATCH_SIZE = 32
STEPS_PER_EPOCH = TRAIN_COUNT // BATCH_SIZE
VALIDATION_STEPS = TEST_COUNT // BATCH_SIZE
WIDTH = 299
HEIGHT = 299
LEARNING_RATE = 0.0001

# Define directories
TRAIN_DIR = 'E:\Attempt 6\Original'
TEST_DIR = 'E:\Attempt 6\Test'
Attempt = "Attempt-1"
filepath_epoch = R"E:\Attempt 6/InceptionFineTuneProper-"+Attempt+"-NoBatchNormalization-" + str(BATCH_SIZE) + "-{epoch:02d}-.model"

# Model used is vgg16 with imagenet weights by default
# Remove output layer as we are replacing it with out own
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
print(len(base_model.layers))

# Declare checkpoints
now = datetime.now()
date_time = now.strftime("%m-%d-%Y")

model_checkpoint = k.callbacks.callbacks.ModelCheckpoint(filepath_epoch, monitor='val_acc', verbose=1, save_best_only=False,
                                                   mode='max')

csv_logger = CSVLogger('finetuning_model_history - ' + date_time + '.csv', append=True)

for layer in base_model.layers:
    layer.trainable = False

# Training new model
x = Sequential()
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)

# we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Create output layer and add output layer to model.
predictions = Dense(CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

print(len(model.layers))
print(len(model.weights))
print(len(model.trainable_weights))

opt = k.optimizers.RMSprop(learning_rate=LEARNING_RATE)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train ImageGen with Aug
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1./255,
    brightness_range=[0.7, 1.3],
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=45,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    horizontal_flip=True,
    fill_mode='nearest')

# Val ImageGen
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1./255,
)

# Train gen
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Validation gen
validation_generator = validation_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# If an image is unreadable drop the batch
# Done to prevent corruption from causing crashes
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

reducelr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4, verbose=1, mode='auto', min_lr=0.00001, countdown=1)

history = model.fit_generator(
    my_gen(train_generator),
    epochs=EPOCHS,
    callbacks=[csv_logger, reducelr, model_checkpoint],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=my_gen(validation_generator),
    shuffle=True,
    validation_steps=VALIDATION_STEPS,
    class_weight=class_weight,
    verbose=1)

# Display history of training at once.
display(history.history)

# Save history as csv file to come back to later
SAVE = 'Newest history' + Attempt + '.csv'
pandas.DataFrame.from_dict(history.history).to_csv(SAVE, index=False)



