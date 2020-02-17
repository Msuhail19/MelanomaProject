import tensorflow as tf
import keras as k




# Choose file to convert
saved_model_dir = './saved_model.pb'
model = tf.keras.models.load_model(saved_model_dir)

# Define converter function and save converter file
converter = tf.lite.TFLiteConverter.from_keras_model('saved_model.pb')
tflite_model = converter.convert()

open("converted_model.lite", "wb").write(tflite_model)