from glob import glob
from sklearn.model_selection import train_test_split
import os
import shutil

# This file is for data preparation
# First seperate cat and dog images randomly into train and test
# Create train and test directories

melanoma = glob('E:/PetImages/train/Cat/*.jpg')
benign = glob('E:/PetImages/train/Dog/*.jpg')

melanoma_train, melanoma_test = train_test_split(melanoma, test_size=0.30)
benign_train, benign_test = train_test_split(benign, test_size=0.30)


# Create new Directory test
test_dir = 'E:/PetImages/test'
os.mkdir(test_dir)

# Create new Directory test/cat
melanoma_dir = 'E:/PetImages/test/Cat'
os.mkdir(melanoma_dir)

print('Moving Malignant Images')

# Move cat test files to you know where
for img in melanoma_test:
    print('.' , end =" ")
    shutil.move(img, melanoma_dir)

# New directory test/dog
benign_dir = 'E:/PetImages/test/Dog'
os.mkdir(benign_dir)

print('Moving Benign Images')

# Move dog images
for img in benign_test:
    print('.' , end =" ")
    shutil.move(img, benign_dir)