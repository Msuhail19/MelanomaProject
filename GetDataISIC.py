from glob import glob
from sklearn.model_selection import train_test_split
import os
import shutil

dir = 'E:/PetImages/test/Dog'
IMAGE_PATH = 'C:/Users/Msuha/OneDrive/Pictures/Melanoma Pictures/Images'
BENIGN_PATH = 'C:/Users/Msuha/OneDrive/Pictures/Melanoma Pictures/FinalDirectory/Benign'
MALIGNANT_PATH = 'C:/Users/Msuha/OneDrive/Pictures/Melanoma Pictures/FinalDirectory/Malignant'

# select all json files relevant
benign_json = glob(BENIGN_PATH + "/*.json")
maglinant_json = glob(MALIGNANT_PATH + "/*.json")

count = 0
list_of_names_malignant = []
print("Select file names")
for b in maglinant_json:
    value = b.strip('C:/Users/Msuha/OneDrive/Pictures/Melanoma Pictures/FinalDirectory/Malignant\\')
    newv = value.strip('.j')
    list_of_names_malignant.append(newv)
print(list_of_names_malignant)


print("Begin finding images")
img_file = []
dest = 'C:/Users/Msuha/OneDrive/Pictures/Melanoma Pictures/FinalDirectory/Malignant Images/'
# Use each new file to find corresponding image, and
for name in list_of_names_malignant:
    path = IMAGE_PATH + "/" + name + ".*"
    img_file = glob(path)
    result = shutil.move(img_file[0], dest)
    print("Moved " + result)

