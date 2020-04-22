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
MODEL_FILE = R'C:\PycharmProjects\Github Repositories\MelanomaProject\Models Gen 2\ClassWeightsAttempt\Inception3-HIGHLR-LOWDROP-BATCH 32-25-.model'

# Load model
loaded_model = load_model(MODEL_FILE)

# Directories of images to test against
benign_imgs = glob(R'E:\Final Generated\Validation\Naevus/*')
mal_imgs = glob(R'E:\Final Generated\Validation\Melanoma/*')

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
        if preds[1] > preds[0]:
            print('Benign')
            benign_no = benign_no + 1
        else:
            print('Malignant')
            mal_no = mal_no + 1

    return benign_no, mal_no

import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    print("Begin plotting matrix")
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    plt.savefig('Confusion Matrix.png')

# Get true positive and false negatives here
False_Benign, True_Malignant = iterate(mal_imgs)
print('False Benign : ' + str(False_Benign))
print('True Mal : ' + str(True_Malignant))

True_Benign, False_Malignant = iterate(benign_imgs)
print('True Benign : ' + str(True_Benign))
print('False Mal : ' + str(False_Malignant))

array = np.array([[True_Malignant, False_Malignant],[False_Benign, True_Benign]])

plot_confusion_matrix(cm = array,  normalize = False, target_names=['Malignant','Benign'], title="Confusion Matrix Over Validation data")