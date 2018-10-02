from __future__ import print_function
import tensorflow as tf
import numpy as np
from auxiliary import progress_bar
import os
import sys


def train(**keywords):
    """
    This function run the session whether in training or evaluation mode.
    NOTE: **keywords is defined in order to make the code easily changable.
    WARNING: All the arguments for the **keywords must be defined when calling this function.
    **keywords:
    :param sess: The default session.
    :param saver: The saver operator to save and load the model weights.
    :param tensors: The tensors dictionary defined by the graph.
    :param data: The data structure.
    :param train_dir: The training dir which is a reference for saving the logs and model checkpoints.
    :param finetuning: If fine tuning should be done or random initialization is needed.
    :param num_epochs: Number of epochs for training.
    :param online_test: If the testing is done while training.
    :param checkpoint_dir: The directory of the checkpoints.
    :param batch_size: The training batch size.

    :return:
             Run the session.
    """

    # The prefix for checkpoint files
    checkpoint_prefix = 'model'

    ###################################################################
    ########## Defining the summary writers for train/test ###########
    ###################################################################

    train_summary_dir = os.path.join(keywords['train_dir'], "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir)
    train_summary_writer.add_graph(keywords['sess'].graph)

    test_summary_dir = os.path.join(keywords['train_dir'], "summaries", "test")
    test_summary_writer = tf.summary.FileWriter(test_summary_dir)
    test_summary_writer.add_graph(keywords['sess'].graph)

    # If fie-tuning flag in 'True' the model will be restored.
    if keywords['finetuning']:
        keywords['saver'].restore(keywords['sess'], os.path.join(keywords['checkpoint_dir'], checkpoint_prefix))
        print("Model restored for fine-tuning...")

    ###################################################################
    ########## Run the training and loop over the batches #############
    ###################################################################
    for epoch in range(keywords['num_epochs']):
        total_batch_training = int(keywords['data'].train.images.shape[0] / keywords['batch_size'])

        # go through the batches
        for batch_num in range(total_batch_training):
            #################################################
            ########## Get the training batches #############
            #################################################

            start_idx = batch_num * keywords['batch_size']
            end_idx = (batch_num + 1) * keywords['batch_size']

            # Fit training using batch data
            train_batch_data, train_batch_label = keywords['data'].train.images[start_idx:end_idx], keywords[
                                                                                                        'data'].train.labels[
                                                                                                    start_idx:end_idx]

            ########################################
            ########## Run the session #############
            ########################################

            # Run optimization op (backprop) and Calculate batch loss and accuracy
            # When the tensor tensors['global_step'] is evaluated, it will be incremented by one.
            batch_loss, _, train_summaries, training_step = keywords['sess'].run(
                [keywords['tensors']['cost'], keywords['tensors']['train_op'], keywords['tensors']['summary_train_op'],
                 keywords['tensors']['global_step']],
                feed_dict={keywords['tensors']['image_place']: train_batch_data,
                           keywords['tensors']['label_place']: train_batch_label,
                           keywords['tensors']['dropout_param']: 0.5})

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
        summary_epoch_train_op = keywords['tensors']['summary_epoch_train_op']
        train_epoch_summaries = keywords['sess'].run(summary_epoch_train_op,
                                                     feed_dict={keywords['tensors']['image_place']: train_batch_data,
                                                                keywords['tensors']['label_place']: train_batch_label,
                                                                keywords['tensors']['dropout_param']: 1.0})

        # Put the summaries to the train summary writer.
        train_summary_writer.add_summary(train_epoch_summaries, global_step=training_step)

        #####################################################
        ########## Evaluation on the test data #############
        #####################################################

        if keywords['online_test']:
            # WARNING: In this evaluation the whole test data is fed. In case the test data is huge this implementation
            #          may lead to memory error. In presense of large testing samples, batch evaluation on testing is
            #          recommended as in the training phase.
            test_accuracy_epoch, test_summaries = keywords['sess'].run(
                [keywords['tensors']['accuracy'], keywords['tensors']['summary_test_op']],
                feed_dict={keywords['tensors']['image_place']: keywords['data'].test.images,
                           keywords['tensors'][
                               'label_place']: keywords['data'].test.labels,
                           keywords['tensors'][
                               'dropout_param']: 1.})
            print("Epoch " + str(epoch + 1) + ", Testing Accuracy= " + \
                  "{:.5f}".format(test_accuracy_epoch))

            ###########################################################
            ########## Write the summaries for test phase #############
            ###########################################################

            # Returning the value of global_step if necessary
            current_step = tf.train.global_step(keywords['sess'], keywords['tensors']['global_step'])

            # Add the couter of global step for proper scaling between train and test summuries.
            test_summary_writer.add_summary(test_summaries, global_step=current_step)

    ###########################################################
    ############ Saving the model checkpoint ##################
    ###########################################################

    # # The model will be saved when the training is done.

    # Create the path for saving the checkpoints.
    if not os.path.exists(keywords['checkpoint_dir']):
        os.makedirs(keywords['checkpoint_dir'])

    # save the model
    save_path = keywords['saver'].save(keywords['sess'], os.path.join(keywords['checkpoint_dir'], checkpoint_prefix))
    print("Model saved in file: %s" % save_path)


    ############################################################################
    ########## Run the session for pur evaluation on the test data #############
    ############################################################################


def evaluation(**keywords):
    # The prefix for checkpoint files
    checkpoint_prefix = 'model'

    # Get the input arguments
    saver = keywords['saver']
    sess = keywords['sess']
    checkpoint_dir = keywords['checkpoint_dir']
    data = keywords['data']
    accuracy_tensor = keywords['tensors']['accuracy']
    image_place = keywords['tensors']['image_place']
    label_place = keywords['tensors']['label_place']
    dropout_param = keywords['tensors']['dropout_param']


    # Restoring the saved weights.
    saver.restore(sess, os.path.join(checkpoint_dir, checkpoint_prefix))
    print("Model restored...")

    test_set = data.test.images
    test_label = data.test.labels
    # Evaluation of the model
    test_accuracy = 100 * keywords['sess'].run(accuracy_tensor, feed_dict={
        image_place: test_set,
        label_place: test_label,
        dropout_param: 1.})

    print("Final Test Accuracy is %% %.2f" % test_accuracy)
