import keras as k
import pandas
import tensorflow
from IPython.core.display import display
from keras import Sequential
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import Model, load_model
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

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
filepath_epoch = R"E:\Attempt 6/InceptionFineTune-"+Attempt+"-Phase2!-" + str(BATCH_SIZE) + "-{epoch:02d}-.model"

def getModel():
    # Load a model and compile ready for retraining the training layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
    x = Sequential()
    for layer in base_model.layers:
        x.add(layer)
    x.add(Dense(512, activation='relu'))
    x.add(Dropout(0.5))

    x.add(Dense(128, activation='relu'))
    x.add(Dropout(0.5))

    # Here we have the output layer
    x.add(Dense(2, activation='sigmoid'))
    return x


new_model = getModel()
new_model = load_model(R'E:\Attempt 6/InceptionFineTune-Attempt-1-NoBatchNormalization-32-03-.model')
new_model.summary()
print(len(new_model.layers))

for layer in new_model.layers:
    layer.trainable = True
for layer in new_model.layers[2:]:
    print(layer)
    layer.trainable = False
#
# opt = k.optimizers.RMSprop(learning_rate=LEARNING_RATE)
# new_model.compile(optimizer=opt,
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# print(len(new_model.trainable_weights))
#
# # Train ImageGen with Aug
# train_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     rescale=1./255,
#     brightness_range=[0.7, 1.3],
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     rotation_range=45,
#     shear_range=0.2,
#     zoom_range=0.2,
#     vertical_flip=True,
#     horizontal_flip=True,
#     fill_mode='nearest')
#
# # Val ImageGen
# validation_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     rescale=1./255,
# )
#
# # Train gen
# train_generator = train_datagen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=(HEIGHT, WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical')
#
# # Validation gen
# validation_generator = validation_datagen.flow_from_directory(
#     TEST_DIR,
#     target_size=(HEIGHT, WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical')
#
# # If an image is unreadable drop the batch
# # Done to prevent corruption from causing crashes
# def my_gen(gen):
#     while True:
#         try:
#             data, labels = next(gen)
#             yield data, labels
#         except:
#             pass
#
#
# # Declare history checkpoint
# csv_logger = CSVLogger('vgg16_phase2_history - .csv', append=True)
#
# # Begin fitting model
# class_weight = {0: 5.,
#                 1: 1.}
#
# model_checkpoint = k.callbacks.callbacks.ModelCheckpoint(filepath_epoch, monitor='val_acc', verbose=1, save_best_only=False,
#                                                    mode='max')
#
# reducelr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4, verbose=1, mode='auto', min_lr=0.00001, countdown=1)
#
# history = new_model.fit_generator(
#     my_gen(train_generator),
#     epochs=EPOCHS,
#     callbacks=[csv_logger, reducelr, model_checkpoint],
#     steps_per_epoch=STEPS_PER_EPOCH,
#     validation_data=my_gen(validation_generator),
#     shuffle=True,
#     validation_steps=VALIDATION_STEPS,
#     class_weight=class_weight,
#     verbose=1)