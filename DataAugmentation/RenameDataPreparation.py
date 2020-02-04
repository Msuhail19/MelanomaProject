from glob import glob
from sklearn.model_selection import train_test_split
import os
import shutil

path = 'E:/SkinDirectory/Train/Naevus\\'


files_to_rename = glob('E:/SkinDirectory/Train/Naevus/*')

# File rename file to a number
count = 0
for file in files_to_rename:
    new_file = file.strip(path)
    ending = ''
    if new_file.endswith('.png'):
        ending = '.png'
    if new_file.endswith('.jpg'):
        ending = '.jpg'
    if new_file.endswith('.jpeg'):
        ending = '.jpeg'

    new_file = new_file.strip(ending)
    os.rename(file, path + str(count) + ending)
    count += 1

