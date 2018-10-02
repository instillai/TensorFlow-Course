import numpy as np
import collections


class DATA_OBJECT(object):
    def __init__(self,
                 images,
                 labels,
                 num_classes=0,
                 one_hot=False,
                 dtype=np.float32,
                 reshape=False):
        """Data object construction.
         images: The images of size [num_samples, rows, columns, depth].
         labels: The labels of size [num_samples,]
         num_classes: The number of classes in case one_hot labeling is desired.
         one_hot=False: Turn the labels into one_hot format.
         dtype=np.float32: The data type.
         reshape=False: Reshape in case the feature vector extraction is desired.

        """
        # Define the date type.
        if dtype not in (np.uint8, np.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_samples = images.shape[0]

        # [num_examples, rows, columns, depth] -> [num_examples, rows*columns]
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])

        # Conver to float if necessary
        if dtype == np.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(dtype)
            images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels

        # If the one_hot flag is true, then the one_hot labeling supersedes the normal labeling.
        if one_hot:
            # If the one_hot labeling is desired, number of classes must be defined as one of the arguments of DATA_OBJECT class!
            assert num_classes != 0, (
                'You must specify the num_classes in the DATA_OBJECT for one_hot label construction!')

            # Define the indexes.
            index = np.arange(self._num_samples) * num_classes
            one_hot_labels = np.zeros((self._num_samples, num_classes))
            one_hot_labels.flat[index + labels.ravel()] = 1
            self._labels = one_hot_labels

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_samples(self):
        return self._num_samples


def provide_data(mnist):
    """
    This function provide data object with desired shape.
    The attribute of data object:
        - train
        - validation
        - test
    The sub attributs of the data object attributes:
        -images
        -labels

    :param mnist: The downloaded MNIST dataset
    :return: data: The data object.
                   ex: data.train.images return the images of the dataset object in the training set!


    """
    ################################################
    ########## Get the images and labels############
    ################################################

    # Note: This setup is specific to mnist data but can be generalized for any data.
    # The ?_images(? can be train, validation or test) must have the format of [num_samples, rows, columns, depth] after extraction from data.
    # The ?_labels(? can be train, validation or test) must have the format of [num_samples,] after extraction from data.
    train_images = mnist.train.images
    train_labels = mnist.train.labels
    validation_images = mnist.validation.images
    validation_labels = mnist.validation.labels
    test_images = mnist.test.images
    test_labels = mnist.test.labels

    # Create separate objects for train, validation & test.
    train = DATA_OBJECT(train_images, train_labels, num_classes=10, one_hot=True, dtype=np.float32, reshape=False)
    validation = DATA_OBJECT(validation_images, validation_labels, num_classes=10, one_hot=True, dtype=np.float32,
                             reshape=False)
    test = DATA_OBJECT(test_images, test_labels, num_classes=10, one_hot=True, dtype=np.float32, reshape=False)

    # Create the whole data object
    DataSetObject = collections.namedtuple('DataSetObject', ['train', 'validation', 'test'])
    data = DataSetObject(train=train, validation=validation, test=test)

    return data
