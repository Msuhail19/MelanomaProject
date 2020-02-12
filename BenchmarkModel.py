from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import keras
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
# from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping

TRAIN_DIR = 'E:/Generate/Train'
TEST_DIR = 'E:/Generate/Test'
BATCH_SIZE = 32
STEPS_PER_EPOCH = 13

base_model = InceptionV3(weights='imagenet', include_top=False)
print('loaded model')

data_gen_args = dict(preprocessing_function=preprocess_input,  # Define the dictionary for Image data Generator
                    rotation_range=30,
                    brightness_range=[0.6, 1.4],
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=True)

# Create data generator
train_datagen = image.ImageDataGenerator(**data_gen_args)
test_datagen = image.ImageDataGenerator(**data_gen_args)

# Create data generator data
train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size=(299, 299), batch_size=BATCH_SIZE)

valid_generator = test_datagen.flow_from_directory(TEST_DIR,
                                                   target_size=(299, 299), batch_size=BATCH_SIZE)

benchmark = Sequential()
benchmark.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(299, 299, 3)))
benchmark.add(MaxPooling2D(pool_size=2, padding='same'))
benchmark.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
benchmark.add(MaxPooling2D(pool_size=2, padding='same'))
benchmark.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
benchmark.add(MaxPooling2D(pool_size=2, padding='same'))
benchmark.add(Dropout(0.3))
benchmark.add(Flatten())
benchmark.add(Dense(512, activation='relu'))
benchmark.add(Dropout(0.5))
benchmark.add(Dense(2, activation='softmax'))
benchmark.summary()

opt = keras.optimizers.RMSprop(learning_rate=0.003, decay=0.9, epsilon=0.1)
benchmark.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Save the model with best weights
checkpointer = ModelCheckpoint('benchmark.model', verbose=1, save_best_only=True)

# Stop the training if the model shows no improvement
stopper = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=0, verbose=1, mode='auto')


# If an image is unreadable drop the batch
def my_gen(gen):
    while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except:
            pass


# Generate
history = benchmark.fit_generator(my_gen(train_generator), steps_per_epoch=13, validation_data=my_gen(valid_generator),
                                  validation_steps=3, epochs=10, verbose=1, callbacks=[checkpointer])
