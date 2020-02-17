import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from glob import glob

# Dimensions of image
HEIGHT = 299
WIDTH = 299

# Model directory
MODEL_FILE = 'Models/FineTuning Inception/82 Trainable Weights, 74% val acc/Inception3-BATCH 32-03-.model'

# Load model
loaded_model = load_model(MODEL_FILE)

# Directories of images to test against
benign_imgs = glob('E:/Totally Independant/Benign Nevi No sticker/*')
benign_imgs2 = glob('E:/Totally Independant/Benign Nevi/*')
mal_imgs = glob('E:/Totally Independant/Melanoma Aug/*')
mal_imgs2 = glob('E:/Totally Independant/Melanoma Internet/*')

# Method to predict and return predictions
def predict(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = loaded_model.predict(x)
    return preds[0]

# Method to take a list of images return no of benign and malignant
def iterate(list):
    print(len(list))
    benign_no = 0
    mal_no = 0
    count = 0
    for item in list:
        img = image.load_img(item, target_size=(HEIGHT, WIDTH))
        preds = predict(img)
        keras.backend.clear_session()
        count = count + 1
        print(str(count) + ' ' + str(preds[0]) + ' ' + str(preds[1]))
        print(item)
        if preds[1] > 0.65:
            print('Benign')
            benign_no = benign_no + 1
        else:
            print('Malignant')
            mal_no = mal_no + 1

    return benign_no, mal_no

# For each image dir
print('Predicting Malignant Images 2 ... ')
ben_no, mal_no = iterate(mal_imgs2)
acc4 = mal_no/len(mal_imgs2)
print('Accuracy is ' + str(acc4))

print('Predicting Malignant Images ... ')
ben_no, mal_no = iterate(mal_imgs)
acc3 = mal_no/len(mal_imgs)
print('Accuracy is ' + str(acc3))

print('Predicting Benign ...')
ben_no , mal_no = iterate(benign_imgs)
acc1 = ben_no/len(benign_imgs)
print('Accuracy is ' + str(acc1))

print('Predicting Benign 2 ...')
ben_no, mal_no = iterate(benign_imgs2)
acc2 = ben_no/len(benign_imgs2)
print('Accuracy is ' + str(acc2))


print('Malignant acc in order is : ' + str(acc3) + ' , ' + str(acc4))
print('Benign acc in order is : ' + str(acc1) + ' , ' + str(acc2))


