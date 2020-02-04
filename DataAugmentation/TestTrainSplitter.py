from glob import glob
from sklearn.model_selection import train_test_split
import os
import shutil

# This file is for data preparation
# First seperate mole images randomly into train and test
# Create train and test directories

melanoma = glob('E:/SkinDirectory/Train/Melanoma/*')
benign = glob('E:/SkinDirectory/train/naevus/*')
test_dir = 'E:/SkinDirectory/Test'
benign_dir = 'E:/SkinDirectory/test/Naevus'
melanoma_dir = 'E:/SkinDirectory/test/Melanoma'

melanoma_train, melanoma_test = train_test_split(melanoma, test_size=0.30)
benign_train, benign_test = train_test_split(benign, test_size=0.10)

# Create new Directory test
os.mkdir(test_dir)

# Create new Directory test - melanoma
os.mkdir(melanoma_dir)

print('Moving Malignant Images')

# Move cat test files to you know where
for img in melanoma_test:
    print('.' , end =" ")
    shutil.move(img, melanoma_dir)

# New directory test - naevus
os.mkdir(benign_dir)

print('Moving Benign Images')

# Move dog images
for img in benign_test:
    print('.' , end =" ")
    shutil.move(img, benign_dir)