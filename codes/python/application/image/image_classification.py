# Import python libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import tensorflow as tf
import random
import pandas as pd
import tensorflow_datasets as tfds
from collections import defaultdict
from skimage.transform import resize
from skimage.util import img_as_float
from skimage import io

""" # Params
"""
TRAIN_LEN = 1000
TEST_LEN = 1000

print(tf.__version__)

# Download the dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

# Get the files by having the url
# Ref: https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file
data_dir = tf.keras.utils.get_file(origin=dataset_url, cache_subdir=os.path.expanduser('~/data'), fname='flower_photos',
                                   untar=True)

# Create a Path object
# Ref: https://docs.python.org/3/library/pathlib.html
data_dir = pathlib.Path(data_dir)

# Get all image paths
image_paths = list(data_dir.glob('*/*.jpg'))

# Create a dataFrame
df = pd.DataFrame(image_paths, columns=['path'])


def get_class(path):
    """
    Get the class labels from the file paths
    :param path: The full path of the file
    :return:
    """
    return path.parent.name


def get_look_up_dict(df):
    """
    Create a look up tables for class labels and their associated unique keys
    :param df: dataframe
    :return: Dict
    """
    # Defining a dict
    look_up_dict = defaultdict(list)
    classes = list(df['class_name'].unique())

    for i in range(len(classes)):
        look_up_dict[classes[i]] = i

    return look_up_dict


# Store the class names in a new column
df['class_name'] = df.path.apply(get_class)

# Create a class to label dictionary
class_to_label = get_look_up_dict(df)
label_to_class = dict([(value, key) for key, value in class_to_label.items()])

# Store the class labels in a new column
df['label'] = df.class_name.apply(lambda x: class_to_label[x])

# Create separate train/test splits
from sklearn.model_selection import train_test_split

X, y = df['path'], df['label']
# Read more at https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# Stratify sampling is used.
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=42, shuffle=True)


def imResize(image):
    """
    This function resize the images.
    :param image: The stack of images
    :return: The stack of resized images
    """
    # Desired size
    IM_SIZE = 200

    # Turn to float64 and scale to [0,1]
    image = img_as_float(image)

    desired_size = [IM_SIZE, IM_SIZE]
    image_resized = resize(image, (desired_size[0], desired_size[1]),
                           anti_aliasing=True)

    # Cast back to float 32
    image_resized = image_resized.astype(np.float32)
    return image_resized


def visualize_training():
    im_list = []
    n_samples_to_show = 9
    c = 0
    for i in range(n_samples_to_show):
        sample, label = next(train_gen())
        im_list.append(sample)
    # Visualization
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure(figsize=(4., 4.))
    # Ref: https://matplotlib.org/3.1.1/gallery/axes_grid1/simple_axesgrid.html
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    # Show image grid
    for ax, im in zip(grid, im_list):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
    plt.show()


""" # Dataset generator
"""


def train_gen():
    """
    The generator function to create training samples
    :return: Generator object
    ex: For next sample use next(train_gen()).
    To loop through:
    gen_obj = train_gen()
    for item in gen_obj:
        print(item)
    """
    for i in range(TRAIN_LEN):
        # Pick a random choice
        idx = np.random.randint(0, TRAIN_LEN)
        im_path = X_train.iloc[idx]
        im_label = y_train.iloc[idx]

        # Read the image
        im = io.imread(str(im_path))

        # Resize the image
        im = imResize(im)

        yield im, im_label


def test_gen():
    """
    The generator function to create test samples
    :return: Generator object
    """
    for i in range(TEST_LEN):
        # Pick a random choice
        idx = np.random.randint(0, TRAIN_LEN)
        im_path = X_train.iloc[idx]
        im_label = y_train.iloc[idx]

        # Read the image
        im = io.imread(str(im_path))

        # Resize the image
        im = imResize(im)

        yield im, im_label


# Get the generator object
sample, label = next(train_gen())

""" # Visualize some sample images from the training set
"""
visualize_training()

""" # Create datasets
"""
batch_size = 32
# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_generator(generator=train_gen, output_types=(tf.float64, tf.uint8))
train_dataset = train_dataset.batch(batch_size)

# Prepare the validation dataset.
test_dataset = tf.data.Dataset.from_generator(generator=test_gen, output_types=(tf.float64, tf.uint8))
test_dataset = test_dataset.batch(batch_size)

# Another way of visualization
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("float32"))
        plt.title(label_to_class[labels[i].numpy()])
        plt.axis("off")
    plt.show()
