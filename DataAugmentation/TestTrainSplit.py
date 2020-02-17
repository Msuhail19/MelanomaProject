from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil

# This file is for data preparation
# First seperate mole images randomly into train and test
# Create train and test directories

melanoma = glob('E:/Generate 2/Train/Melanoma/*')
benign = glob('E:/Generate 2/Train/naevus/*')
test_dir = 'E:/Generate 2/Test'
benign_dir = 'E:/Generate 2/Test/Naevus'
melanoma_dir = 'E:/Generate 2/Test/Melanoma'

melanoma_train, melanoma_test = train_test_split(melanoma, test_size=0.20, random_state=np.random)
benign_train, benign_test = train_test_split(benign, test_size=0.20, random_state=np.random)

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