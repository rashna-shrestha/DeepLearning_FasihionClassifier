#load images from folders

from tensorflow.keras.applications.xception import preprocess_input # this is used to preprocess the images
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img #ImageDataGenerator makes batches, load_img opens one file
import config
# import warnings
# warnings.filterwarnings("ignore")

AUGMENTATION = dict(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=10.0,
    zoom_range=0.2,
    vertical_flip=True
) #AUGMENTATION means we will apply these transformations to the images, which will make the model more robust to variations in the images

def make_train_ds(input_size=config.INPUT_SIZE,batch_size=config.BATCH_SIZE,augmentations=None):
    gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        **(augmentations or {})
    )
    return gen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(input_size, input_size),
        batch_size=batch_size
    )

def make_evaluation_ds(directory, input_size=config.INPUT_SIZE, batch_size=config.BATCH_SIZE):
    gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    return gen.flow_from_directory(
        directory,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        shuffle=False
    )


def load_image_batch(path, input_size=config.INPUT_SIZE):
    img = load_img(path, target_size=(input_size, input_size))
    X = np.array([np.array(img)], dtype="float32")
    return preprocess_input(X)
