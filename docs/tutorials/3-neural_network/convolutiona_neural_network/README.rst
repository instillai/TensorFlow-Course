==============================================
Convolutional Neural Networks using TensorFlow
==============================================

This tutorial deals with training a classifier using convolutional
neural networks. The source code is available at `this link <https://github.com/astorfi/TensorFlow-World/tree/master/codes/3-neural_networks/convolutional-neural-network/>`_.

------------
Introduction
------------


In this tutorial, we try to teach you how to implement a simple neural
network image classifier using **Convolutional Neural Networks(CNNs)**.
The main goal of this post is to show hot to train a CNN classifier
using `TensorFlow <https://www.tensorflow.org/>`__ deep learning
framework developed by Google. The deep learning concepts such as the
details of CNNs will not be discussed here. In order to get a better
idea of convolutional layers and realize how the work please refer to
`this
post <http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html>`__.
In the next section, we start to describe procedure of learning the
classifier.

--------------
Input Pipeline
--------------

The dataset that we work on that in this tutorial is the
`MNIST <http://yann.lecun.com/exdb/mnist/>`__ dataset probably the most
famous dataset in computer vision because of its simplicity! The main
dataset consists of 60000 training and 10000 test images. However, there
might be different setups for these images. The one we use is the same
in the test set but we split the training set to 55000 images as train
and 5000 images as the validation set in the case that using
cross-validation for determining some hyper-parameters is desired. The
images are 28x28x1 which each of them represent a hand-written digit
from 0 to 9. Since this tutorial is supposed to be ready-to-use, we
provided the code to download and extract the MNIST data as a data
object. Thanks to TensorFlow its code is already written and is ready to
use and its source code is available at `this
repository <tensorflow.contrib.learn.python.learn.datasets.mnist>`__ .
The code for downloading and extracting MNIST dataset is as is as below:

.. code:: python

       from tensorflow.examples.tutorials.mnist import input_data
       import tensorflow as tf
       mnist = input_data.read_data_sets("MNIST_data/", reshape=False, one_hot=False)
       data = input.provide_data(mnist)


The above code download and extract MNIST data in the MNIST\_data/
folder in the current directory that we are running the python script.
The reshape flag is set to **False** because we want the image format as
it is which is 28x28x1. The reason is that we are aimed to train a
CNN classifier which takes images as input. If the one\_hot flag is set
to **True** it returns class labels as a one\_hot label. However, we set
the one\_hot flag to **False** for customized preprocessing and data
organization. The **input.provide\_data** function is provided to get
any data with specific format separated by training and testing sets and
return the structured data object for further processing. From now on we
consider **data** as the data object.

In any of the train, validation and test attributes, sub-attributes of
images and labels exist. The have just not been depicted for the
simplicity of the above chart presentation. As an example if
**data.train.imege** is called its shape is
[number\_of\_training\_sample,28,28,1]. It is recommended to play around
a little bit with the data object to grasp a better idea of how it works and
what is its output. The codes are available in the GitHub repository for
this post.

--------------------
Network Architecture
--------------------

After explanation of the data input pipeline, Now it's the time to go
through the neural network architecture used for this tutorial. The
implemented architecture is very similar to
`LeNet <http://yann.lecun.com/exdb/lenet/>`__ although our architecture
is implemented in a fully-convolutional fashion, i.e., there is no
fully-connected layer and all fully-connected layers are transformed to
corresponding convolutional layers. In order to grasp a better idea of
how to go from a fully-connected layer to a convolutional one and vice
verse please refer to `this
link <http://cs231n.github.io/convolutional-networks/>`__. The general
architecture schematic is as below:

.. figure:: https://github.com/astorfi/TensorFlow-World/blob/master/docs/_img/3-neural_network/convolutiona_neural_network/architecture.png
   :scale: 50 %
   :align: center

   **Figure 1:** The general architecture of the network.

   
The image is depicted by
`Tensorboard <https://www.tensorflow.org/get_started/summaries_and_tensorboard>`__
as a visualization tool for TensorFlow. Later on in this tutorial, the
way of using Tensorboard and make the most of it will be explained. As
it can be seen by the figure, the convolutional layers are followed by
pooling layers and the last fully-connected layer is followed by a
dropout layer to decrease the overfitting. *The dropout will only be
applied in the training phase*. The code for designing the architecture
is as below:

.. code:: python

    import tensorflow as tf
    slim = tf.contrib.slim

    def net_architecture(images, num_classes=10, is_training=False,
                         dropout_keep_prob=0.5,
                         spatial_squeeze=True,
                         scope='Net'):

        # Create empty dictionary
        end_points = {}

        with tf.variable_scope(scope, 'Net', [images, num_classes]) as sc:
            end_points_collection = sc.name + '_end_points'

            # Collect outputs for conv2d and max_pool2d.
            with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.max_pool2d], 
            outputs_collections=end_points_collection):
            
                # Layer-1
                net = tf.contrib.layers.conv2d(images, 32, [5,5], scope='conv1')
                net = tf.contrib.layers.max_pool2d(net, [2, 2], 2, scope='pool1')

                # Layer-2
                net = tf.contrib.layers.conv2d(net, 64, [5, 5], scope='conv2')
                net = tf.contrib.layers.max_pool2d(net, [2, 2], 2, scope='pool2')

                # Layer-3
                net = tf.contrib.layers.conv2d(net, 1024, [7, 7], padding='VALID', scope='fc3')
                net = tf.contrib.layers.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout3')

                # Last layer which is the logits for classes
                logits = tf.contrib.layers.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='fc4')

                # Return the collections as a dictionary
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                # Squeeze spatially to eliminate extra dimensions.
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='fc4/squeezed')
                    end_points[sc.name + '/fc4'] = logits
                return logits, end_points
 
    def net_arg_scope(weight_decay=0.0005):
        #Defines the default network argument scope.

        with tf.contrib.framework.arg_scope(
                [tf.contrib.layers.conv2d],
                padding='SAME',
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                                   uniform=False, seed=None,
                                                                                   dtype=tf.float32),
                activation_fn=tf.nn.relu) as sc:
            return sc

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Default Parameters and Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function net\_arg\_scope is defined to share some attributes between
layers. It is very useful in the cases which some attributes like 'SAME'
padding (which is zero-padding in essence) are joint between different
layer. It basically does the sharing variable with some pre-definitions.
Basically, it enables us to specify different operations and/or a set of
arguments to be passed to any of the defined operations in the
arg\_scope. So for this specific case the argument
**tf.contrib.layers.conv2d** is defined and so all the convolutional
layers default parameters (which are set by the arg\_scope) are as
defined in the arg\_scope. The is more work to use this useful
arg\_scope operation and it will be explained in the general TensorFlow
implementation details later on in this tutorial. It is worth noting
that all the parameters defined by arg\_scope, can be overwritten
locally in the specific layer definition. As an example take a look at defining the tf.contrib.layers.conv2d layer(the
convolutional layer), the padding is set to **'VALID'** although its
default been set to **'SAME'** by the arg\_scope operation. Now it's the
time to explain the architecture itself by describing of how to create
convolutional and pooling layers.

ReLU has been used as the non-linear activation function for all the
layers except for the last layer(embedding layer). The famous Xavier
initialization has not been used for initialization of the network and
instead, the Variance-Scaling-Initializer has been used which provided
more promising results in the case of using ReLU activation. The
advantage is to keep the scale of the input variance constant, so it is
claimed that it does not explode or diminish by getting to the final
layer\ `[reference] <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/variance_scaling_initializer>`__.
There are different types of variance-scaling initializers. The one we
used in is the one proposed by the paper `Understanding the difficulty
of training deep feedforward neural
networks <http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf>`__
and provided by the TensorFlow. is the one proposed by the paper
`Understanding the difficulty of training deep feedforward neural
networks <http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf>`__
and provided by the TensorFlow.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Convolution and Pooling Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now it's the time to build our convolutional architecture using
convolution and pooling layers which are defined in the
net\_architecture panel in the above python script. It is worth noting
that since the output of layers(output tensors) are different by the
size the output sizes decrease gradually as we go through the depth of
the network, the matching between inputs-outputs of the layers must be
considered and at the end, the output of the last layer should be transformed
into a feature vector in order to be fed to the embedding layer.

Defining pooling layers is straightforward as it is shown. The defined pooling layer has the kernel size of 2x2 and a stride
of 2 in each dimension. This is equivalent to extract the maximum in
each 2x2 windows and the stride makes no overlapping in the chosen
windows for max pooling operation. In order to have a better
understanding of pooling layer please refer to `this
link <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/max_pool2d>`__.

Convolution layers can be defined using
`tf.contrib.layers <https://www.tensorflow.org/api_docs/python/tf/contrib/layers>`__.
The default padding is set to 'SAME' as mentioned before. loosely
speaking, 'SAME' padding equals to same spatial dimensions for output
feature map and input feature map which contains zero padding to
matching the shapes and theoretically, it is done equally on every side
of the input map. One the other hand, 'VALID' means no padding. The
overall architecture of the convolution layer is as depicted below:

.. figure:: https://github.com/astorfi/TensorFlow-World/blob/master/docs/_img/3-neural_network/convolutiona_neural_network/convlayer.png
   :scale: 30 %
   :align: center
       
   **Figure 2:** The operations in the convolutional layer.


The number of **output feature maps** is set to 32 and the **spatial kernel size** is set to [5,5]. The
**stride** is [1,1] by default. The **scope** argument is for defining
the name for the layer which is useful in different scenarios such as
returning the output of the layer, fine-tuning the network and graphical
advantages like drawing a nicer graph of the network using Tensorboard.
Basically, it is the representative of the layer and adds all the
operations into a higher-level node.

We overwrote the padding type. It is changed to
'VALID' padding. The reason is behind the characteristics of the
convolutional layer. It is operating as a
fully-connected layer. *It is not because of the 'VALID' padding
though*. The 'VALID' padding is just part of the mathematical operation.
The reason is that the input to this layer has the spatial size of
**7x7** and the kernel size of the layer is the same. This is obvious
because when the input size of the convolutional layer equals to its
kernel size and 'VALID' pooling is used, the output is only one single
neuron if the number of output feature map equals to 1. So if the number
of output feature maps is equals to 1024, this layer operates like and
filly-connected layer with 1024 output hidden units!

~~~~~~~~~~~~~
Dropout Layer
~~~~~~~~~~~~~

The dropout is one of the most famous methods in order to prevent
over-fitting. This operation randomly kills a portion of the neurons to
stochastically force the neurons to learn more useful information.
The method is stochastic and it's been widely used in neural
network architecture and presented promising results. The dropout\_keep\_prob argument determines
the portion of the neurons which remains untouched and will not be
disabled by the dropout layer. Moreover, the flag is\_training is
supposed to affect the dropout layer and force it to be **active** in the training phase and **deactive** in
the test/evaluation phase.

~~~~~~~~~~~~~~~
Embedding Layer
~~~~~~~~~~~~~~~

A Convolutional layer results in a 4-dimensional tensor with the dimensionality of
 [batch\_size, width, height, channel]. As a result, the embedding layer
combines all the channels except the first one indicating the batches.
So the dimension of [batch\_size, width, height, channel] becomes
[batch\_size, width x height x channel]. This
is the last fully-connected layer prior to softmax which the number of
its output units must be equal to the number of classes. The output of
this layer has the dimensionality of [batch\_size, 1, 1, num\_classes].
The ``tf.squeeze`` function does the embedding operation which its output dimension
is [batch\_size, num\_classes]. It is worth noting that the scope of the
last layer overwrites the scope='fc4'.

--------------------
The TensorFlow Graph
--------------------

At this time, after describing the network design and different layers,
it is the time to present how to implement this architecture using
TensorFlow. With TensorFlow everything should be defined on something
called GRAPH. The graphs have the duty to tell the TensorFlow backend to
what to do and how to do the desired operations. TensorFlow uses Session
to run the operations.

The graph operations are executed in session environment which contains
state of variables. For running each created session a specific graph is
needed because each session can only be operated on a single graph. So
multiple graphs cannot be used in a single session. If the users do
not explicitly use a session by its name, the default session will be
used by TensorFlow.

A graph contains tensors and the operations defined on that graph. So
the graph can be used on multiple sessions. Again like the sessions, if
a graph is not explicitly defined by the user, the TensorFlow itself sets
a default graph. Although there is no harm working with the default
graph, but explicitly defining the graph is recommended. The general
graph of out experimental setup is as below:

.. figure:: https://github.com/astorfi/TensorFlow-World/blob/master/docs/_img/graph.png
   :scale: 30 %
   :align: center

   **Figure 3:** The TensorFlow Graph.



The graph is explicitly defined in our experiments. The following
script, panel by panel, shows the graph design of our experiments:

.. code:: python
     
    graph = tf.Graph()
    with graph.as_default():

        # global step
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # learning rate policy
        decay_steps = int(num_train_samples / FLAGS.batch_size *
                          FLAGS.num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   global_step,
                                                   decay_steps,
                                                   FLAGS.learning_rate_decay_factor,
                                                   staircase=True,
                                                   name='exponential_decay_learning_rate')


        # Place holders
        image_place = tf.placeholder(tf.float32, shape=([None, height, width, num_channels]), name='image')
        label_place = tf.placeholder(tf.float32, shape=([None, FLAGS.num_classes]), name='gt')
        dropout_param = tf.placeholder(tf.float32)

     
        # MODEL
        arg_scope = net.net_arg_scope(weight_decay=0.0005)
        with tf.contrib.framework.arg_scope(arg_scope):
            logits, end_points = net.net_architecture(image_place, num_classes=FLAGS.num_classes, dropout_keep_prob=dropout_param,
                                           is_training=FLAGS.is_training)

        # Define loss
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_place))

        # Accuracy
        with tf.name_scope('accuracy'):
            # Evaluate model
            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_place, 1))

            # Accuracy calculation
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

     
        # Define optimizer by its default values
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Gradient update.
        with tf.name_scope('train'):
            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

     
        arr = np.random.randint(data.train.images.shape[0], size=(3,))
        tf.summary.image('images', data.train.images[arr], max_outputs=3,
                         collections=['per_epoch_train'])

        # Histogram and scalar summaries sammaries
        for end_point in end_points:
            x = end_points[end_point]
            tf.summary.scalar('sparsity/' + end_point,
                              tf.nn.zero_fraction(x), collections=['train', 'test'])
            tf.summary.histogram('activations/' + end_point, x, collections=['per_epoch_train'])

        # Summaries for loss, accuracy, global step and learning rate.
        tf.summary.scalar("loss", loss, collections=['train', 'test'])
        tf.summary.scalar("accuracy", accuracy, collections=['train', 'test'])
        tf.summary.scalar("global_step", global_step, collections=['train'])
        tf.summary.scalar("learning_rate", learning_rate, collections=['train'])

        # Merge all summaries together.
        summary_train_op = tf.summary.merge_all('train')
        summary_test_op = tf.summary.merge_all('test')
        summary_epoch_train_op = tf.summary.merge_all('per_epoch_train')


Each of the above sections will be explained in the following subsections
using the same naming convention for convenience.

~~~~~~~~~~~~~
Graph Default
~~~~~~~~~~~~~

As mentioned before, it is recommended to set the graph manually and in
that section, we named the graph to be **graph**. Later on, it will 
notice that this definition is useful because we can pass the graph to
other functions and sessions and it will be recognized.

~~~~~~~~~~
Parameters
~~~~~~~~~~

Different parameters are necessary for the learning procedure. The
global\_step is one of which. The reason behind
defining the global\_step is to have a track of where we are in the
training procedure. It is a non-learnable tensor and should be
incremented per each gradient update which will be done over each batch.
The decay\_steps determines after how many steps
or epochs the learning rate should be decreased by a predefined policy.
As can be seen **num\_epochs\_per\_decay** defines the decay factor
which is restricted to the number of passed epochs. The learning\_rate
tensor determines the learning rate policy.
Please refer to TensorFlow official documentation for grasping a better
idea of the *tf.train.exponential\_decay* layer. It is worth noting that
the *tf.train.exponential\_decay* layer takes *global\_step* as its
counter to realize when it has to change the learning rate.

~~~~~~~~~~~~~
Place Holders
~~~~~~~~~~~~~

The tf.placeholder operation creates a placeholder variable tensor
which will be fed to the network in testing/training phase. The images
and labels must have placeholders because they are in essence the inputs
to the network in training/testing phase. The *type* and *shape* of the
place-holders must be defined as required parameters. The first dimension of the shape argument is set to
**None** which allows the place holder to get any dimension. The first
dimension is the *batch\_size* and is flexible.

The dropout\_param placeholder takes the probability of keeping a
neuron active. The reason behind defining a place-holder for the dropout
parameter is to enable the setup to take this parameter in running each
any session arbitrary which enrich the experiment to disable it when
running the testing session.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model and Evaluation Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default provided parameters are determined by
**arg\_scope** operator. The
*tf.nn.softmax\_cross\_entropy\_with\_logits* is operating on the un-normalized
logits as the loss function. This function computes the softmax
activation internally which makes it more stable. Finally, the accuracy is computed.

~~~~~~~~~~~~~~~~
Training Tensors
~~~~~~~~~~~~~~~~

Now it's the time to define the training tensors. The Adam Optimizer is used as one of the best current optimization
algorithms and has been utilized widely used because of its adaptive
characteristics and outstanding performance. The gradients must
be computed using the *defined loss tensor* and those computations must
be added as the *train operations* to the graph. Basically 'train\_op'
is an operation that is run for gradient update on parameters. Each
execution of 'train\_op' is a training step. By passing 'global\_step'
to the optimizer, each time that the 'train\_op' is run, TensorFlow
update the 'global\_step' and increment it by one!

~~~~~~~~~
Summaries
~~~~~~~~~

In this section, we describe how to create summary operations and save
them into allocated tensors. Eventually, the summaries should be
presented in *Tensorboard* in order to visualize what is happening
inside of the network black-box. There are different types of summaries.
Three type of image, scalar and histogram summaries are used in this
implementations. In order to avoid this post to becoming too verbose, we
do not go in depth of the explanation for summary operations and we will
get back to it in another post.

The image summaries are created which has the duty of
visualizing the input elements to the summary tensor. These elements here
are 3 random images from the train data. In The outputs of different layers will be fed to the relevant summary tensor.
Finally, some scalar summaries are created in order
to track the *training convergence* and *testing performance*. The
collections argument in summary definitions is a supervisor which direct
each summary tensor to the relevant operation. For example, some
summaries only need to be generated in training phase and some are only
needed in testing. We have a collection named 'per\_epoch\_train' too
and the summaries which only have to be generated once per epoch in the
training phase will be stored in this list. Eventually, the summaries are gathered in the
corresponding summary lists using the collections key.

--------
Training
--------

Now it's the time to go through the training procedure. It consists of
different steps which start by **session configuration** to saving the
**model checkpoint**.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configuration and Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First of all the tensors should be gathered for convenience and the
session must be configured. The code is as below:

.. code:: python

     
    tensors_key = ['cost', 'accuracy', 'train_op', 'global_step', 'image_place', 'label_place', 'dropout_param',
                       'summary_train_op', 'summary_test_op', 'summary_epoch_train_op']
    tensors = [loss, accuracy, train_op, global_step, image_place, label_place, dropout_param, summary_train_op,
                   summary_test_op, summary_epoch_train_op]
    tensors_dictionary = dict(zip(tensors_key, tensors))

    # Configuration of the session
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(graph=graph, config=session_conf)


As it is clear, all the tensors are stored in a
dictionary to be used later by the corresponding keys. The allow\_soft\_placement
flag allows the switching back-and-forth between different devices.
This is useful when the user allocated 'GPU' to all operations without
considering the fact that not all operations are supported by GPU using
the TensorFlow. In this case, if the *allow\_soft\_placement* operator is
disabled, errors can block the program to continue and the user must start the debugging
process but the usage of the active flag prevent this issue by automatically
switch from a non-supported device to the supported one. The
log\_device\_placement flag is to present which operations are set on
what devices. This is useful for debugging and it projects a verbose
dialog in the terminal. Eventually, the session is taken
using the defined **graph**. The training phase starts using the
following script:

.. code:: python

     
    with sess.as_default():
        # Run the saver.
        # 'max_to_keep' flag determine the maximum number of models that the tensorflow save and keep. default by TensorFlow = 5.
        saver = tf.train.Saver(max_to_keep=FLAGS.max_num_checkpoint)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        ###################################################
        ############ Training / Evaluation ###############
        ###################################################
        train_evaluation.train(sess, saver, tensors_dictionary, data,
                                 train_dir=FLAGS.train_dir,
                                 finetuning=FLAGS.fine_tuning,
                                 num_epochs=FLAGS.num_epochs, checkpoint_dir=FLAGS.checkpoint_dir,
                                 batch_size=FLAGS.batch_size)
                                     
        train_evaluation.evaluation(sess, saver, tensors_dictionary, data,
                               checkpoint_dir=FLAGS.checkpoint_dir)


The tf.train.Saver is run in order to provide an
operation to save and load the models. The **max\_to\_keep** flags
determine the maximum number of the saved models that the TensorFlow
keeps and its default is set to '5' by TensorFlow. The
session is run in order to initialize all the variable which is
necessary. Finally, train\_evaluation function is
provided to run the training/testing phase.

~~~~~~~~~~~~~~~~~~~
Training Operations
~~~~~~~~~~~~~~~~~~~

The training function is as below:

.. code:: python

     
    from __future__ import print_function
    import tensorflow as tf
    import numpy as np
    import progress_bar
    import os
    import sys

    def train(sess, saver, tensors, data, train_dir, finetuning,
                    num_epochs, checkpoint_dir, batch_size):
        """
        This function run the session whether in training or evaluation mode.
        :param sess: The default session.
        :param saver: The saver operator to save and load the model weights.
        :param tensors: The tensors dictionary defined by the graph.
        :param data: The data structure.
        :param train_dir: The training dir which is a reference for saving the logs and model checkpoints.
        :param finetuning: If fine tuning should be done or random initialization is needed.
        :param num_epochs: Number of epochs for training.
        :param checkpoint_dir: The directory of the checkpoints.
        :param batch_size: The training batch size.

        :return:
                 Run the session.
        """

        # The prefix for checkpoint files
        checkpoint_prefix = 'model'

        ###################################################################
        ########## Defining the summary writers for train /test ###########
        ###################################################################

        train_summary_dir = os.path.join(train_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir)
        train_summary_writer.add_graph(sess.graph)

        test_summary_dir = os.path.join(train_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir)
        test_summary_writer.add_graph(sess.graph)

        # If fie-tuning flag in 'True' the model will be restored.
        if finetuning:
            saver.restore(sess, os.path.join(checkpoint_dir, checkpoint_prefix))
            print("Model restored for fine-tuning...")

        ###################################################################
        ########## Run the training and loop over the batches #############
        ###################################################################
        for epoch in range(num_epochs):
            total_batch_training = int(data.train.images.shape[0] / batch_size)

            # go through the batches
            for batch_num in range(total_batch_training):
                #################################################
                ########## Get the training batches #############
                #################################################

                start_idx = batch_num * batch_size
                end_idx = (batch_num + 1) * batch_size

                # Fit training using batch data
                train_batch_data, train_batch_label = data.train.images[start_idx:end_idx], data.train.labels[
                                                                                            start_idx:end_idx]

                ########################################
                ########## Run the session #############
                ########################################

                # Run optimization op (backprop) and Calculate batch loss and accuracy
                # When the tensor tensors['global_step'] is evaluated, it will be incremented by one.
                batch_loss, _, train_summaries, training_step = sess.run(
                    [tensors['cost'], tensors['train_op'], tensors['summary_train_op'],
                     tensors['global_step']],
                    feed_dict={tensors['image_place']: train_batch_data,
                               tensors['label_place']: train_batch_label,
                               tensors['dropout_param']: 0.5})

                ########################################
                ########## Write summaries #############
                ########################################

                # Write the summaries
                train_summary_writer.add_summary(train_summaries, global_step=training_step)

                # # Write the specific summaries for training phase.
                # train_summary_writer.add_summary(train_image_summary, global_step=training_step)

                #################################################
                ########## Plot the progressive bar #############
                #################################################

                progress = float(batch_num + 1) / total_batch_training
                progress_bar.print_progress(progress, epoch_num=epoch + 1, loss=batch_loss)

            # ################################################################
            # ############ Summaries per epoch of training ###################
            # ################################################################
            train_epoch_summaries = sess.run(tensors['summary_epoch_train_op'],
                                             feed_dict={tensors['image_place']: train_batch_data,
                                                        tensors['label_place']: train_batch_label,
                                                        tensors['dropout_param']: 0.5})

            # Put the summaries to the train summary writer.
            train_summary_writer.add_summary(train_epoch_summaries, global_step=training_step)

            #####################################################
            ########## Evaluation on the test data #############
            #####################################################

            # WARNING: In this evaluation the whole test data is fed. In case the test data is huge this implementation
            #          may lead to memory error. In presence of large testing samples, batch evaluation on testing is
            #          recommended as in the training phase.
            test_accuracy_epoch, test_summaries = sess.run([tensors['accuracy'], tensors['summary_test_op']],
                                                           feed_dict={tensors['image_place']: data.test.images,
                                                                      tensors[
                                                                          'label_place']: data.test.labels,
                                                                      tensors[
                                                                          'dropout_param']: 1.})
            print("Epoch " + str(epoch + 1) + ", Testing Accuracy= " + \
                  "{:.5f}".format(test_accuracy_epoch))

            ###########################################################
            ########## Write the summaries for test phase #############
            ###########################################################

            # Returning the value of global_step if necessary
            current_step = tf.train.global_step(sess, tensors['global_step'])

            # Add the counter of global step for proper scaling between train and test summaries.
            test_summary_writer.add_summary(test_summaries, global_step=current_step)

        ###########################################################
        ############ Saving the model checkpoint ##################
        ###########################################################

        # # The model will be saved when the training is done.

        # Create the path for saving the checkpoints.
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # save the model
        save_path = saver.save(sess, os.path.join(checkpoint_dir, checkpoint_prefix))
        print("Model saved in file: %s" % save_path)


        ############################################################################
        ########## Run the session for pur evaluation on the test data #############
        ############################################################################
    def evaluation(sess, saver, tensors, data, checkpoint_dir):

            # The prefix for checkpoint files
            checkpoint_prefix = 'model'

            # Restoring the saved weights.
            saver.restore(sess, os.path.join(checkpoint_dir, checkpoint_prefix))
            print("Model restored...")

            # Evaluation of the model
            test_accuracy = 100 * sess.run(tensors['accuracy'], feed_dict={tensors['image_place']: data.test.images,
                                                                           tensors[
                                                                               'label_place']: data.test.labels,
                                                                           tensors[
                                                                               'dropout_param']: 1.})

            print("Final Test Accuracy is %% %.2f" % test_accuracy)


The input parameters to the function are described in the comments. The summary writers are defined
separately for train and test phases. The program
checks if fine-tuning is desired then the model is loaded and the
operation will be continued afterward. The batches
are extracted from training data. For a single
training step, the model is evaluated on a batch of data and the model
parameter and weights will be updated. The model finally will be
saved.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Training Summaries and Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The training loop saves the summaries in the train summary part. By
using the Tensorboard and pointing to the directory that the logs are
saved, we can visualize the training procedure. The loss and accuracy
for the train are depicted jointly as below:

.. figure:: https://github.com/astorfi/TensorFlow-World/blob/master/docs/_img/3-neural_network/convolutiona_neural_network/loss_accuracy_train.png
   :scale: 50 %
   :align: center
   
   **Figure 4:** The loss and accuracy curves for training.


The activation of the last fully-connected layer will be depicted in the
following figure:

.. figure:: https://github.com/astorfi/TensorFlow-World/blob/master/docs/_img/3-neural_network/convolutiona_neural_network/activation_fc4_train.png
   :scale: 50 %
   :align: center
   
   **Figure 5:** The activation of the last layer.


For the last layer, it is good to have a visualization of the
distribution of the neurons outputs. By using the histogram summary the
distribution can be shown over the whole training steps. The result is
as below:

.. figure:: https://github.com/astorfi/TensorFlow-World/blob/master/docs/_img/3-neural_network/convolutiona_neural_network/histogram_fc4_train.png
   :scale: 50 %
   :align: center
   
   **Figure 6:** The histogram summary of the last layer.


Eventually, the test accuracy per step is plotted as the following curve:

.. figure:: https://github.com/astorfi/TensorFlow-World/blob/master/docs/_img/3-neural_network/convolutiona_neural_network/test_accuracy.png
   :scale: 50 %
   :align: center

   **Figure 7:** Test Accuracy.



A representation of the terminal progressive bar for the training phase
is as below:

.. figure:: https://github.com/astorfi/TensorFlow-World/blob/master/docs/_img/3-neural_network/convolutiona_neural_network/terminal_training.png
   :scale: 50 %
   :align: center
   
   **Figure 8:** The terminal in training phase.



Few things need to be considered in order to clarify the results:

-  The initial learning rate by the **Adam optimizer** has been set to a
   small number. By setting that to a larger number, the speed of
   increasing the accuracy could go higher.However, this is not always the case. We deliberately set that to a
   small number to be able to track the procedure easier.
-  The **histogram summaries** are saved per each epoch and not per
   step. Since the generation of histogram summaries is very
   time-consuming, there are only generated per epoch of training.
-  While the training is under process, per each epoch, an evaluation
   will be performed over the whole test set. If the test set is too
   big, isolated evaluation is recommended in order to avoid the memory
   exhaustion issue.

-------
Summary
-------

In this tutorial, we train a neural network classifier using
convolutional neural networks. MNIST data has been used for simplicity
and its wide usage. The TensorFlow has been used as the deep learning
framework. The main goal of this tutorial was to present an easy
ready-to-use implementation of training classifiers using TensorFLow.
Lots of the tutorials in this category looks like to be too verbose in
code or too short in explanations. My effort was to provide a tutorial
to be easily understandable in the sense of coding and be comprehensive
in the sense of description. Some of the details about some
TensorFlow(like summaries) and data-input-pipeline have been ignored for
simplicity. We get back to them in the future posts. I hope you enjoyed
it.

