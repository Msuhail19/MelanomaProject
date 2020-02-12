from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from glob import glob
import random
import math
from PIL import Image
import shutil

img_to_save = sorted(glob('E:/NewDataSet/Melanoma PPM/*'))
save_dir = 'E:/NewDataSet/Melanoma/'

count = 0
for image in img_to_save:
    img = Image.open(image)
    img.save(save_dir + 'GEN' + str(count) + '.jpg')
    count +=1