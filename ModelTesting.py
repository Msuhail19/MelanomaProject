import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from glob import glob
import tensorboard


HEIGHT = 299
WIDTH = 299
MODEL_FILE = 'FineTuning Inception/187 Trainable Weights/Inception3-BATCH 16-0515BRIGHTNESS-09-.model'
test_dir = 'E:/SkinDirectory/Train'
benign_imgs = glob('E:/Original/Independent/Naevus No Sticker/*')
mal_imgs = glob('E:/Original/Independent/Melanoma/*')

loaded_model = load_model(MODEL_FILE)
loaded_model.is_trainable = True

def predict(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = loaded_model.predict(x)
    return preds[0]

benign_no = 0
mal_no = 0
count = 0
print('Predicting Malignant')
print(len(mal_imgs))
for mal in mal_imgs:
    img = image.load_img(mal, target_size=(HEIGHT, WIDTH))
    preds = predict(img)
    keras.backend.clear_session()
    count = count + 1
    print(str(count) + ' ' +  str(preds[0]) + ' ' + str(preds[1]))
    print(mal)
    if preds[1] > preds[0]:
        print('Benign')
        benign_no = benign_no + 1
    elif preds[0] > preds[1]:
        print('Malignant')
        mal_no = mal_no + 1


print('\nFinal Mal accuracy is :', end='')
mal_acc = mal_no/len(mal_imgs)
print(mal_acc)



print('Predicting Benign')
benign_no = 0
mal_no = 0
count = 0
for benign in benign_imgs:
    img = image.load_img(benign, target_size=(HEIGHT, WIDTH))
    preds = predict(img)
    keras.backend.clear_session()
    count = count + 1
    print(str(count) + ' ' +  str(preds[0]) + ' ' + str(preds[1]))
    print(benign)
    # Order of probabilities is mal (preds[0]) then benign (preds[1])
    if preds[1] > preds[0]:
        print('Benign')
        benign_no = benign_no + 1
    elif preds[0] > preds[1]:
        print('Malignant')
        mal_no = mal_no + 1

print('\nFinal Benign accuracy is :', end='')
if len(benign_imgs) is not 0 :
    benign_acc = benign_no/len(benign_imgs)
else :
    benign_acc = 'N/A'
print(benign_acc)




print('Overall : ' + str(benign_acc) + ' %benign acc')
print('Overall : ' + str(mal_acc) + ' %mal acc')



def test(pic):
    img = image.load_img(pic, target_size=(HEIGHT, WIDTH))
    preds = predict(img)
    keras.backend.clear_session()
    print(str(count) + ' ' + str(preds[0]) + ' ' + str(preds[1]))
    print(mal)
    if preds[1] > preds[0]:
        print('Benign')
    elif preds[0] > preds[1]:
        print('Malignant')