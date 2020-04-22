import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from glob import glob

# Load each image and run predictions
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=R"C:\PycharmProjects\Github Repositories\MelanomaProject\ConvertedFinalModel.tflite")
interpreter.allocate_tensors()

# Method to predict and return predictions
def process_img(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def predict(data):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    interpreter.set_tensor(input_details[0]['index'], data)

    # Invoke the interpreter
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    return output_data


def iterate(list):
    mal_count = 0
    ben_count = 0
    for pic in list:
        print(pic)
        print(mal_count+ben_count)
        pic = image.load_img(pic, target_size=(299, 299))
        data = process_img(pic)
        output_data = predict(data)
        if output_data[0][0] > output_data[0][1]:
            print('Melanoma')
            mal_count += 1
        else:
            print('Benign')
            ben_count += 1
    return mal_count, ben_count


# Predict
# For each image dir
# Directories of images to test against
benign_imgs = glob(R'E:\Attempt 6\Test\Naevus/*')
benign_imgs2 = glob('E:/Images/Totally Independant/Benign Nevi/*')
mal_imgs = glob(R'E:\Attempt 6\Test\Melanoma/*')
mal_imgs2 = glob('E:/Images/Totally Independant/Melanoma Internet/*')

print('Predicting Malignant Images 2 ... ')
mal_no, ben_no = iterate(mal_imgs2)
acc4 = mal_no / len(mal_imgs2)
print('Accuracy is ' + str(acc4))

print('Predicting Malignant Images ... ')
mal_no, ben_no = iterate(mal_imgs)
acc3 = mal_no / len(mal_imgs)
print('Accuracy is ' + str(acc3))

print('Predicting Benign ...')
mal_no, ben_no = iterate(benign_imgs)
acc1 = ben_no / len(benign_imgs)
print('Accuracy is ' + str(acc1))

print('Predicting Benign 2 ...')
mal_no, ben_no = iterate(benign_imgs2)
acc2 = ben_no / len(benign_imgs2)
print('Accuracy is ' + str(acc2))

print('Malignant acc in order is : ' + str(acc3) + ' , ' + str(acc4))
print('Benign acc in order is : ' + str(acc1) + ' , ' + str(acc2))
