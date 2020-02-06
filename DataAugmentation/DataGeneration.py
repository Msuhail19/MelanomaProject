from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from glob import glob
import random
import math
from PIL import Image
import shutil

# Folder to generate large number of images

img_to_aug = sorted(glob('E:/SkinDirectory/Train/naevus/*'))
print(img_to_aug)
save_dir = 'E:/SkinDirectory/Generate/Generated/naevus'  # save gen files to following dir
# times per image to gen randomised version
rotated_to_create = 2
no_files_transform = 0


# THE FOLLOWING CODE TAKES A RECTANGLE, ROTATES IT
#  THEN FINDS MAX AREA OF NEW RECANGLE THAT IS UPRIGHT
#  IGNORE FOR NOW
def rotatedRectWithMaxArea(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


#
# Code to generate rotated and cropped images

# Iterate over existing and rotate and crop
count = 0
for image in img_to_aug:
    for x in range(rotated_to_create):
        # Open image, generate random angle to rotate and call rotatedRectWithMaxArea
        # to get new height and width
        img = Image.open(image)
        rand = random.randint(0, 359)
        print('rotated ' + str(rand))
        width, height = img.size
        new_width, new_height = rotatedRectWithMaxArea(width, height, math.radians(rand))
        rotated = img.rotate(rand, expand=True)

        # Save the rotated and print rotated dimensions
        print(str(new_width) + ',' + str(new_height))

        # Calculate coordinates to crop
        width, height = rotated.size
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2

        # crop and save
        if x != 1:
            rotated1 = rotated.crop((left, top, right, bottom))
        else:
            rotated1 = img

        try:
            rotated1.save(save_dir + '/GEN' + str(count) + '-RC-' + str(x) + '-' + str(rand) + '.jpg')
        except:
            rotated1.save(save_dir + '/GEN' + str(count) + '-RC-' + str(x) + '-' + str(rand) + '.png')
    count += 1

datagen = ImageDataGenerator(
    brightness_range=[0.4, 1.65],
    horizontal_flip=True,
    fill_mode='reflect', )

print('create new')
images = glob(save_dir + '/*')
print(len(images))

# If we are supposed to randomly transform then do just that
count = 0
if no_files_transform > 0:
    for image in images:
        x = load_img(image)
        x = img_to_array(x)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, save_to_dir=save_dir, save_prefix='NEW' + str(count) + '-' + str(i) + 'NEW',
                                  save_format='png',
                                  batch_size=1):
            if i >= no_files_transform:
                break
            i += 1
        count += 1
