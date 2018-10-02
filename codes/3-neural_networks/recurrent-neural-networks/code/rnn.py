'''
MNIST using Recurrent Neural Network to predict handwritten digits
In this tutorial, I am going to demonstrate how to use recurrent neural 
network to predict the famous handwritten digits "MNIST".
The MNIST dataset consists:
mnist.train: 55000 training images
mnist.validation: 5000 validation images
mnist.test: 10000 test images
Each image is 28 pixels (rows) by 28 pixels (cols).
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Useful function for arguments.
def str2bool(v):
    return v.lower() in ("yes", "true")

# Parser
parser = argparse.ArgumentParser(description='Creating Classifier')

######################
# Optimization Flags #
######################

parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--seed', default=111, type=int, help='seed')

##################
# Training Flags #
##################
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
parser.add_argument('--num_epoch', default=10, type=int, help='Number of training iterations')
parser.add_argument('--batch_per_log', default=10, type=int, help='Print the log at what number of batches?')

###############
# Model Flags #
###############
parser.add_argument('--hidden_size', default=128, type=int, help='Number of neurons for RNN hodden layer')

# Add all arguments to parser
args = parser.parse_args()


# Reset the graph set the random numbers to be the same using "seed"
tf.reset_default_graph()
tf.set_random_seed(args.seed)
np.random.seed(args.seed)

# Divide 28x28 images to rows of data to feed to RNN as sequantial information
step_size = 28
input_size = 28
output_size = 10

# Input tensors
X = tf.placeholder(tf.float32, [None, step_size, input_size])
y = tf.placeholder(tf.int32, [None])

# Rnn
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=args.hidden_size)
output, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# Forward pass and loss calcualtion
logits = tf.layers.dense(state, output_size)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

# Prediction
prediction = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

# input data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

# Process MNIST
X_test = mnist.test.images # X_test shape: [num_test, 28*28]
X_test = X_test.reshape([-1, step_size, input_size])
y_test = mnist.test.labels

# initialize the variables
init = tf.global_variables_initializer()

# Empty list for tracking
loss_train_list = []
acc_train_list = []

# train the model
with tf.Session() as sess:
    sess.run(init)
    n_batches = mnist.train.num_examples // args.batch_size
    for epoch in range(args.num_epoch):
        for batch in range(n_batches):
            X_train, y_train = mnist.train.next_batch(args.batch_size)
            X_train = X_train.reshape([-1, step_size, input_size])
            sess.run(optimizer, feed_dict={X: X_train, y: y_train})
        loss_train, acc_train = sess.run(
            [loss, accuracy], feed_dict={X: X_train, y: y_train})
        loss_train_list.append(loss_train)
        acc_train_list.append(acc_train)
        print('Epoch: {}, Train Loss: {:.3f}, Train Acc: {:.3f}'.format(
            epoch + 1, loss_train, acc_train))
    loss_test, acc_test = sess.run(
        [loss, accuracy], feed_dict={X: X_test, y: y_test})
    print('Test Loss: {:.3f}, Test Acc: {:.3f}'.format(loss_test, acc_test))
