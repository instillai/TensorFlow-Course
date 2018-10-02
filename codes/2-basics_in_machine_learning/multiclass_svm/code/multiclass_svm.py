import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA

#######################
### Necessary Flags ###
#######################

tf.app.flags.DEFINE_integer('batch_size', 50,
                            'Number of samples per batch.')

tf.app.flags.DEFINE_integer('num_steps', 1000,
                            'Number of steps for training.')

tf.app.flags.DEFINE_integer('log_steps', 50,
                            'Number of steps per each display.')

tf.app.flags.DEFINE_boolean('is_evaluation', True,
                            'Whether or not the model should be evaluated.')

tf.app.flags.DEFINE_float(
    'gamma', -15.0,
    'penalty parameter of the error term.')

tf.app.flags.DEFINE_float(
    'initial_learning_rate', 0.01,
    'The initial learning rate for optimization.')

FLAGS = tf.app.flags.FLAGS


###########################
### Necessary Functions ###
###########################
def cross_class_label_fn(A):
    """
    This function take the matrix of size (num_classes, batch_size) and return the cross-class label matrix
    in which Yij are the elements where i,j are class indices.
    :param A: The input matrix of size (num_classes, batch_size).
    :return: The output matrix of size (num_classes, batch_size, batch_size).
    """
    label_class_i = tf.reshape(A, [num_classes, 1, FLAGS.batch_size])
    label_class_j = tf.reshape(label_class_i, [num_classes, FLAGS.batch_size, 1])
    returned_mat = tf.matmul(label_class_j, label_class_i)
    return returned_mat


# Compute SVM loss.
def loss_fn(alpha, label_placeholder):
    term_1 = tf.reduce_sum(alpha)
    alpha_cross = tf.matmul(tf.transpose(alpha), alpha)
    cross_class_label = cross_class_label_fn(label_placeholder)
    term_2 = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(alpha_cross, cross_class_label)), [1, 2])
    return tf.reduce_sum(tf.subtract(term_2, term_1))


# Gaussian (RBF) prediction kernel
def kernel_pred(x_data, prediction_grid):
    A = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
    B = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
    square_distance = tf.add(tf.subtract(A, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))),
                             tf.transpose(B))
    return tf.exp(tf.multiply(gamma, tf.abs(square_distance)))


def kernel_fn(x_data, gamma):
    """
    This function generates the RBF kernel.
    :param x_data: Input data
    :param gamma: Hyperparamet.
    :return: The RBF kernel.
    """
    square_distance = tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))
    kernel = tf.exp(tf.multiply(gamma, tf.abs(square_distance)))
    return kernel


def prepare_label_fn(label_onehot):
    """
    Label preparation. Since we are dealing with one vs all scenario, for each sample
    all the labels other than the current class must be set to -1. It can be done by simply
    Setting all the zero values to -1 in the return one_hot array for classes.

    :param label_onehot: The input as one_hot label which shape (num_samples,num_classes)
    :return: The output with the same shape and all zeros tured to -1.
    """
    labels = label_onehot
    labels[labels == 0] = -1
    labels = np.transpose(labels)
    return labels


def next_batch(X, y, batch_size):
    """
    Generating a batch of random data.
    :param x_train:
    :param batch_size:
    :return:
    """
    idx = np.random.choice(len(X), size=batch_size)
    X_batch = X[idx]
    y_batch = y[:, idx]
    return X_batch, y_batch


########################
### Data Preparation ###
########################

# Read MNIST data. It has a data structure.
# mnist.train.images, mnist.train.labels: The training set images and their associated labels.
# mnist.validation.images, mnist.validation.labels: The validation set images and their associated labels.
# mnist.test.images, mnist.test.labels: The test set images and their associated labels.

# Flags:
#      "reshape=True", by this flag, the data will be reshaped to (num_samples,num_features)
#      and since each image is 28x28, the num_features = 784
#      "one_hot=True", this flag return one_hot labeling format
#      ex: sample_label [1 0 0 0 0 0 0 0 0 0] says the sample belongs to the first class.
mnist = input_data.read_data_sets("MNIST_data/", reshape=True, one_hot=True)

# Label preparation.
y_train = prepare_label_fn(mnist.train.labels)
y_test = prepare_label_fn(mnist.test.labels)

# Get the number of classes.
num_classes = y_train.shape[0]

##########################################
### Dimensionality Reduction Using PCA ###
##########################################
pca = PCA(n_components=100)
pca.fit(mnist.train.images)

# print the accumulative variance for the returned principle components.
print("The variance of the chosen components = %{0:.2f}".format(100 * np.sum(pca.explained_variance_ratio_)))
x_train = pca.transform(mnist.train.images)
x_test = pca.transform(mnist.test.images)
num_fetures = x_train.shape[1]

############################
### Graph & Optimization ###
############################
# Create graph
sess = tf.Session()

# Initialize placeholders
data_placeholder = tf.placeholder(shape=[None, num_fetures], dtype=tf.float32)
label_placeholder = tf.placeholder(shape=[num_classes, None], dtype=tf.float32)
pred_placeholder = tf.placeholder(shape=[None, num_fetures], dtype=tf.float32)

# The alpha variable for solving the dual optimization problem.
alpha = tf.Variable(tf.random_normal(shape=[num_classes, FLAGS.batch_size]))

# Gaussian (RBF) kernel
gamma = tf.constant(FLAGS.gamma)

# RBF kernel
my_kernel = kernel_fn(data_placeholder, gamma)

# Loss calculation.
loss = loss_fn(alpha, label_placeholder)

# Generating the prediction kernel.
pred_kernel = kernel_pred(data_placeholder, pred_placeholder)

#############################
### Prediction & Accuracy ###
#############################
prediction_output = tf.matmul(tf.multiply(label_placeholder, alpha), pred_kernel)
prediction = tf.arg_max(prediction_output - tf.expand_dims(tf.reduce_mean(prediction_output, 1), 1), 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(label_placeholder, 0)), tf.float32))

# Optimizer
train_op = tf.train.AdamOptimizer(FLAGS.initial_learning_rate).minimize(loss)

# Variables Initialization.
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
for i in range(FLAGS.num_steps):

    batch_X, batch_y = next_batch(x_train, y_train, FLAGS.batch_size)
    sess.run(train_op, feed_dict={data_placeholder: batch_X, label_placeholder: batch_y})

    temp_loss = sess.run(loss, feed_dict={data_placeholder: batch_X, label_placeholder: batch_y})

    acc_train_batch = sess.run(accuracy, feed_dict={data_placeholder: batch_X,
                                                   label_placeholder: batch_y,
                                                   pred_placeholder: batch_X})

    batch_X_test, batch_y_test = next_batch(x_test, y_test, FLAGS.batch_size)
    acc_test_batch = sess.run(accuracy, feed_dict={data_placeholder: batch_X_test,
                                                  label_placeholder: batch_y_test,
                                                  pred_placeholder: batch_X_test})

    if (i + 1) % FLAGS.log_steps == 0:
        print('Step #%d, Loss= %f, training accuracy= %f, testing accuracy= %f ' % (
            (i+1), temp_loss, acc_train_batch, acc_test_batch))
