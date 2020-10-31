from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os

######################################
######### Necessary Flags ############
######################################

tf.app.flags.DEFINE_string(
    'train_root', os.path.dirname(os.path.abspath(__file__)) + '/train_logs',
    'Directory where event logs are written to.')

tf.app.flags.DEFINE_string(
    'checkpoint_root',
    os.path.dirname(os.path.abspath(__file__)) + '/checkpoints',
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_integer('max_num_checkpoint', 10,
                            'Maximum number of checkpoints that TensorFlow will keep.')

tf.app.flags.DEFINE_integer('num_classes', 10,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_integer('batch_size', np.power(2, 7),
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_integer('num_epochs', 5,
                            'Number of epochs for training.')

##########################################
######## Learning rate flags #############
##########################################
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.95, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 1, 'Number of epoch pass to decay learning rate.')

#########################################
########## status flags #################
#########################################
tf.app.flags.DEFINE_boolean('is_training', False,
                            'Training/Testing.')

tf.app.flags.DEFINE_boolean('fine_tuning', False,
                            'Fine tuning is desired or not?.')

tf.app.flags.DEFINE_boolean('online_test', True,
                            'Fine tuning is desired or not?.')

tf.app.flags.DEFINE_boolean('allow_soft_placement', True,
                            'Automatically put the variables on CPU if there is no GPU support.')

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Demonstrate which variables are on what device.')

# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS

################################################
################# handling errors!##############
################################################
if not os.path.isabs(FLAGS.train_root):
    raise ValueError('You must assign absolute path for --train_root')

if not os.path.isabs(FLAGS.checkpoint_root):
    raise ValueError('You must assign absolute path for --checkpoint_root')

##########################################
####### Load and Organize Data ###########
##########################################
'''
In this part the input must be prepared.

   1 - The MNIST data will be downloaded.
   2 - The images and labels for both training and testing will be extracted.
   3 - The prepared data format(?,784) is different by the appropriate image shape(?,28,28,1) which needs
        to be fed to the CNN architecture. So it needs to be reshaped.

'''

# Download and get MNIST dataset(available in tensorflow.contrib.learn.python.learn.datasets.mnist)
# It checks and download MNIST if it's not already downloaded then extract it.
# The 'reshape' is True by default to extract feature vectors but we set it to false to we get the original images.
mnist = input_data.read_data_sets("MNIST_data/", reshape=True, one_hot=True)
train_data = mnist.train.images
train_label = mnist.train.labels
test_data = mnist.test.images
test_label = mnist.test.labels

# # The 'input.provide_data' is provided to organize any custom dataset which has specific characteristics.
# data = input.provide_data(mnist)

# Dimentionality of train
dimensionality_train = train_data.shape

# Dimensions
num_train_samples = dimensionality_train[0]
num_features = dimensionality_train[1]

#######################################
########## Defining Graph ############
#######################################

graph = tf.Graph()
with graph.as_default():
    ###################################
    ########### Parameters ############
    ###################################

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

    ###############################################
    ########### Defining place holders ############
    ###############################################
    image_place = tf.placeholder(tf.float32, shape=([None, num_features]), name='image')
    label_place = tf.placeholder(tf.float32, shape=([None, FLAGS.num_classes]), name='gt')
    dropout_param = tf.placeholder(tf.float32)

    ##################################################
    ########### Model + Loss + Accuracy ##############
    ##################################################

    # MODEL(MPL with two hidden layer)

    # LAYER-1
    net = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs=250, scope='fc-1')

    # LAYER-2
    net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=250, scope='fc-2')

    # SOFTMAX
    logits_pre_softmax = tf.contrib.layers.fully_connected(inputs=net, num_outputs=FLAGS.num_classes, scope='fc-3')

    # Define loss
    softmax_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits_pre_softmax, labels=label_place))

    # Accuracy
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(logits_pre_softmax, 1), tf.argmax(label_place, 1)), tf.float32))

    #############################################
    ########### training operation ##############
    #############################################

    # Define optimizer by its default values
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # 'train_op' is a operation that is run for gradient update on parameters.
    # Each execution of 'train_op' is a training step.
    # By passing 'global_step' to the optimizer, each time that the 'train_op' is run, Tensorflow
    # update the 'global_step' and increment it by one!

    # gradient update.
    with tf.name_scope('train_scope'):
        grads = optimizer.compute_gradients(softmax_loss)
        train_op = optimizer.apply_gradients(grads, global_step=global_step)

    ###############################################
    ############ Define Sammaries #################
    ###############################################

    # Summaries for loss and accuracy
    tf.summary.scalar("loss", softmax_loss, collections=['train', 'test'])
    tf.summary.scalar("accuracy", accuracy, collections=['train', 'test'])
    tf.summary.scalar("global_step", global_step, collections=['train'])
    tf.summary.scalar("learning_rate", learning_rate, collections=['train'])

    # Merge all summaries together.
    summary_train_op = tf.summary.merge_all('train')
    summary_test_op = tf.summary.merge_all('test')

    ############################################
    ############ Run the Session ###############
    ############################################
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(graph=graph, config=session_conf)

    with sess.as_default():
        # Run the saver.
        # 'max_to_keep' flag determines the maximum number of models that the tensorflow save and keep. default by TensorFlow = 5.
        saver = tf.train.Saver(max_to_keep=FLAGS.max_num_checkpoint)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        ###################################################
        ############ Training / Evaluation ###############
        ###################################################

        # The prefix for checkpoint files
        checkpoint_prefix = 'model'

        ###################################################################
        ########## Defining the summary writers for train/test ###########
        ###################################################################

        train_summary_dir = os.path.join(FLAGS.train_root, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir)
        train_summary_writer.add_graph(sess.graph)

        test_summary_dir = os.path.join(FLAGS.train_root, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir)
        test_summary_writer.add_graph(sess.graph)

        # If fine-tuning flag in 'True' the model will be restored.
        if FLAGS.fine_tuning:
            saver.restore(sess, os.path.join(FLAGS.checkpoint_root, checkpoint_prefix))
            print("Model restored for fine-tuning...")

        ###################################################################
        ########## Run the training and loop over the batches #############
        ###################################################################
        for epoch in range(FLAGS.num_epochs):
            total_batch_training = int(train_data.shape[0] / FLAGS.batch_size)

            # go through the batches
            for batch_num in range(total_batch_training):
                #################################################
                ########## Get the training batches #############
                #################################################

                start_idx = batch_num * FLAGS.batch_size
                end_idx = (batch_num + 1) * FLAGS.batch_size

                # Fit training using batch data
                train_batch_data, train_batch_label = train_data[start_idx:end_idx], train_label[
                                                                                     start_idx:end_idx]

                ########################################
                ########## Run the session #############
                ########################################

                # Run optimization op (backprop) and Calculate batch loss and accuracy
                # When the tensor tensors['global_step'] is evaluated, it will be incremented by one.
                batch_loss, _, train_summaries, training_step = sess.run(
                    [softmax_loss, train_op,
                     summary_train_op,
                     global_step],
                    feed_dict={image_place: train_batch_data,
                               label_place: train_batch_label,
                               dropout_param: 0.5})

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

            print("Epoch #" + str(epoch + 1) + ", Train Loss=" + \
                  "{:.3f}".format(batch_loss))

            #####################################################
            ########## Evaluation on the test data #############
            #####################################################

            if FLAGS.online_test:
                # WARNING: In this evaluation the whole test data is fed. In case the test data is huge this implementation
                #          may lead to memory error. In presense of large testing samples, batch evaluation on testing is
                #          recommended as in the training phase.
                test_accuracy_epoch, test_summaries = sess.run(
                    [accuracy, summary_test_op],
                    feed_dict={image_place: test_data,
                               label_place: test_label,
                               dropout_param: 1.})
                print("Test Accuracy= " + \
                      "{:.4f}".format(test_accuracy_epoch))

                ###########################################################
                ########## Write the summaries for test phase #############
                ###########################################################

                # Returning the value of global_step if necessary
                current_step = tf.train.global_step(sess, global_step)

                # Add the couter of global step for proper scaling between train and test summuries.
                test_summary_writer.add_summary(test_summaries, global_step=current_step)

        ###########################################################
        ############ Saving the model checkpoint ##################
        ###########################################################

        # # The model will be saved when the training is done.

        # Create the path for saving the checkpoints.
        if not os.path.exists(FLAGS.checkpoint_root):
            os.makedirs(FLAGS.checkpoint_root)

        # save the model
        save_path = saver.save(sess, os.path.join(FLAGS.checkpoint_root, checkpoint_prefix))
        print("Model saved in file: %s" % save_path)

        ############################################################################
        ########## Run the session for pur evaluation on the test data #############
        ############################################################################

        # The prefix for checkpoint files
        checkpoint_prefix = 'model'

        # Restoring the saved weights.
        saver.restore(sess, os.path.join(FLAGS.checkpoint_root, checkpoint_prefix))
        print("Model restored...")

        # Evaluation of the model
        total_test_accuracy = sess.run(accuracy, feed_dict={
            image_place: test_data,
            label_place: test_label,
            dropout_param: 1.})

        print("Final Test Accuracy is %.2f" % total_test_accuracy)
