from glob import glob
import random
import math
from PIL import Image

# Folder to generate large number of images
img_to_aug = glob(R'E:\Attempt 2\Original\Naevus/*.*')

print(img_to_aug)
# save gen files to following dir
save_dir = R'E:\Attempt 2\Train/Naevus'

# times per image to gen randomised version
rotated_to_create = 2


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

def rotateAndAugmentImage(image, count, x):
    img = Image.open(image)
    rand = random.randint(0, 180)
    print('rotated ' + str(rand))
    width, height = img.size
    new_width, new_height = rotatedRectWithMaxArea(width, height, math.radians(rand))
    rotated = img.rotate(rand, expand=True)

    # Save the rotated and print rotated dimensions
    print(str(new_width) + ',' + str(new_height))

    #rotated.save(save_dir + '/GEN' + str(count) + '-RC-' + str(x) + '-' + str(rand) + '.jpg')

    # Calculate coordinates to crop
    width, height = rotated.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    rotated1 = rotated.crop((left, top, right, bottom))
    print("Saving...")
    try:
        rotated1.save(save_dir + '/GEN2' + str(count) + '-' + str(x) + '.jpg')
    except(Exception):
        rotated1.save(save_dir + '/GEN2' + str(count) + '-' + str(x) + '.png')




# Code to generate rotated and cropped images
# Iterate over existing and rotate and crop
count = 0
for image in img_to_aug:
    # Open image, generate random angle to rotate and call rotatedRectWithMaxArea
    # to get new height and width
    print('Image : ' + str(count))
    while True:
        try:
            rotateAndAugmentImage(image, count, 1)
            break
        except(Exception):
            print(Exception)
            continue

    while True:
        try:
            rotateAndAugmentImage(image, count, 2)
            break
        except(Exception):
            print(Exception)
            continue

    count += 1


