import tensorflow as tf

# Load trainining model save as final model
model_dir = R'E:\Attempt 6\Inception3-0005-Attempt4-NODROPOUT-BATCH 32-25-.model'
old_model = tf.keras.models.load_model(model_dir)

# Converter
converter = tf.lite.TFLiteConverter.from_keras_model(old_model)

# Convert and save
tflite_model = converter.convert()
open("ConvertedFinalModel.tflite", "wb").write(tflite_model)
