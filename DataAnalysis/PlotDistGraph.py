import numpy as np
import keras
import pandas
import pandas as pandas
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from glob import glob

# Importing the figure factory
import plotly.figure_factory as ff
from plotly.offline import iplot
import plotly

# Dimensions of image
HEIGHT = 299
WIDTH = 299

# Model directory
MODEL_FILE = R'E:\Attempt 6\Inception3-0005-Attempt4-NODROPOUT-BATCH 32-25-.model'

# Directories of images to test against
mal_imgs = glob(R'E:\Images\Images by type\Smartphone only\melanoma/*')
benign_imgs = glob(R'E:\Images\Images by type\Smartphone only\naevus/*')

# Load model
loaded_model = load_model(MODEL_FILE)


# Method to predict and return predictions
def predict(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = loaded_model.predict(x)
    return preds[0]

# Method to take a list of images return no of benign and malignant
def iterate(list):
    # Print size of list
    print(len(list))

    # Create list to store prediction results
    lst = []
    for item in list:
        # Load model, predict and clear session
        img = image.load_img(item, target_size=(HEIGHT, WIDTH))
        preds = predict(img)
        keras.backend.clear_session()

        # preds[0] malignant probability, preds[1] is benign probability
        # Round probability to whole number
        lst.append(preds[0]*100)

    # Return list of probabilities
    return lst

# For each image dir
print('Show distribution of malignant smartphone images ')
mal_list = iterate(mal_imgs)
ben_list = iterate(benign_imgs)

hist_data = [mal_list, ben_list]
group_labels = ["Malignant Images", "Benign Images"]
colors = ['blue', 'red']

fig = ff.create_distplot(hist_data=hist_data, group_labels=group_labels, bin_size=[3,3], colors=colors, curve_type='normal')
fig.update_layout(title='Dist Plot Over Malignant Probability, All images')

iplot(fig, filename="C:/PycharmProjects/Github Repositories/MelanomaProject/malignant", image='png')
