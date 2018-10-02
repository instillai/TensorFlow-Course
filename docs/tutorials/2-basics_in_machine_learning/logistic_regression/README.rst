

Sections
~~~~~~~~

-  `Introduction <#Introduction>`__
-  `Description of the Overall
   Process <#Description%20of%20the%20Overall%20Process>`__
-  `How to Do It in Code? <#How%20to%20Do%20It%20in%20Code?>`__
-  `Summary <#Summary>`__

Logistic Regression using TensorFlow
------------------------------------

This tutorial is about training a logistic regression by TensorFlow for
binary classification.

Introduction
------------

In `Linear Regression using
TensorFlow <http://www.machinelearninguru.com/deep_learning/tensorflow/machine_learning_basics/linear_regresstion/linear_regression.html>`__
post we described how to predict continuous-valued parameters by
linearly modeling the system. What if the objective is to decide between
two choices? The answer is simple: we are dealing with a classification
problem. In this tutorial, the objective to decide whether the input
image is digit "0" or digit "1" using Logistic Regression. In another
word, whether it is digit "1" or not! The full source code is available
in the associated `GitHub
repository <https://github.com/Machinelearninguru/Deep_Learning/tree/master/TensorFlow/machine_learning_basics/logistic_regression>`__.

Dataset
-------

The dataset that we work on that in this tutorial is the
`MNIST <http://yann.lecun.com/exdb/mnist/>`__ dataset. The main dataset
consists of 55000 training and 10000 test images. The images are 28x28x1
which each of them represent a hand-written digit from 0 to 9. We create
feature vectors of size 784 of each image. We only use 0 and 1 images
for our setting.

Logistic Regression
-------------------

In linear regression the effort is to predict the outcome continuous
value using the linear function of $y=W^{T}x$. On the other hand, in
logistic regression we are determined to predict a binary label as
$y\\in\\{0,1\\}$ in which we use a different prediction process as
opposed to linear regression. In logistic regression, the predicted
output is the probability that the input sample belongs to a targeted
class which is digit "1" in our case. In a binary-classification
problem, obviously if the $P(x\\in\\{target\\\_class\\})$ = M, then
$P(x\\in\\{non\\\_target\\\_class\\}) = 1 - M$. So the hypothesis can be
created as follows:

$$P(y=1\|x)=h\_{W}(x)={{1}\\over{1+exp(-W^{T}x)}}=Sigmoid(W^{T}x) \\ \\
\\ (1)$$ $$P(y=0\|x)=1 - P(y=1\|x) = 1 - h\_{W}(x) \\ \\ \\ (2)$$

In the above equations, Sigmoid function maps the predicted output into
probability space in which the values are in the range of $[0,1]$. The main
objective is to find the model using which when the input sample is "1"
the output becomes a high probability and becomes small otherwise. The
important objective is to design the appropriate cost function to
minimize the loss when the output is desired and vice versa. The cost
function for a set of data such as $(x^{i},y^{i})$ can be defined as
below:

$$Loss(W) =
\\sum\_{i}{y^{(i)}log{1\\over{h\_{W}(x^{i})}}+(1-y^{(i)})log{1\\over{1-h\_{W}(x^{i})}}}$$

As it can be seen from the above equation, the loss function consists of
two term and in each sample only one of them is non-zero considering the
binary labels.

Up to now, we defined the formulation and optimization function of the
logistic regression. In the next part, we show how to do it in code using
mini-batch optimization.

Description of the Overall Process
----------------------------------

At first, we process the dataset and extract only "0" and "1" digits. The
code implemented for logistic regression is heavily inspired by our
`Train a Convolutional Neural Network as a
Classifier <http://www.machinelearninguru.com/deep_learning/tensorflow/neural_networks/cnn_classifier/cnn_classifier.html>`__
post. We refer to the aforementioned post for having a better
understanding of the implementation details. In this tutorial, we only
explain how we process dataset and how to implement logistic regression
and the rest is clear from the CNN classifier post that we referred
earlier.

How to Do It in Code?
---------------------

In this part, we explain how to extract desired samples from dataset and
to implement logistic regression using Softmax.

Process Dataset
~~~~~~~~~~~~~~~

At first, we need to extract "0" and "1" digits from MNIST dataset:

.. code:: python

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", reshape=True, one_hot=False)

    ########################
    ### Data Processing ####
    ########################
    # Organize the data and feed it to associated dictionaries.
    data={}

    data['train/image'] = mnist.train.images
    data['train/label'] = mnist.train.labels
    data['test/image'] = mnist.test.images
    data['test/label'] = mnist.test.labels

    # Get only the samples with zero and one label for training.
    index_list_train = []
    for sample_index in range(data['train/label'].shape[0]):
        label = data['train/label'][sample_index]
        if label == 1 or label == 0:
            index_list_train.append(sample_index)

    # Reform the train data structure.
    data['train/image'] = mnist.train.images[index_list_train]
    data['train/label'] = mnist.train.labels[index_list_train]


    # Get only the samples with zero and one label for test set.
    index_list_test = []
    for sample_index in range(data['test/label'].shape[0]):
        label = data['test/label'][sample_index]
        if label == 1 or label == 0:
            index_list_test.append(sample_index)

    # Reform the test data structure.
    data['test/image'] = mnist.test.images[index_list_test]
    data['test/label'] = mnist.test.labels[index_list_test]

The code looks to be verbose but it's very simple actually. All we want
is implemented in lines 28-32 in which the desired data samples are
extracted. Next, we have to dig into logistic regression architecture.

Logistic Regression Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The logistic regression structure is simply feeding-forwarding the input
features through a fully-connected layer in which the last layer only
has two classes. The fully-connected architecture can be defined as
below:

.. code:: python

        ###############################################
        ########### Defining place holders ############
        ###############################################
        image_place = tf.placeholder(tf.float32, shape=([None, num_features]), name='image')
        label_place = tf.placeholder(tf.int32, shape=([None,]), name='gt')
        label_one_hot = tf.one_hot(label_place, depth=FLAGS.num_classes, axis=-1)
        dropout_param = tf.placeholder(tf.float32)

        ##################################################
        ########### Model + Loss + Accuracy ##############
        ##################################################
        # A simple fully connected with two class and a Softmax is equivalent to Logistic Regression.
        logits = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs = FLAGS.num_classes, scope='fc')

The first few lines are defining place-holders in order to put the
desired values on the graph. Please refer to `this
post <http://www.machinelearninguru.com/deep_learning/tensorflow/neural_networks/cnn_classifier/cnn_classifier.html>`__
for further details. The desired loss function can easily be implemented
using TensorFlow using the following script:

.. code:: python

        # Define loss
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot))

        # Accuracy
        with tf.name_scope('accuracy'):
            # Evaluate the model
            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))

            # Accuracy calculation
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

The tf.nn.softmax\_cross\_entropy\_with\_logits function does the work.
It optimizes the previously defined cost function with a subtle
difference. It generates two inputs in which even if the sample is digit
"0", the correspondent probability will be high. So
tf.nn.softmax\_cross\_entropy\_with\_logits function, for each class
predict a probability and inherently on its own, makes the decision.

Summary
-------

In this tutorial, we described logistic regression and represented how to
implement it in code. Instead of making a decision based on the output
probability based on a targeted class, we extended the problem to a two
class problem in which for each class we predict the probability. In the future posts, we will extend this problem to multi-class problem and we
show it can be done with the similar approach.
