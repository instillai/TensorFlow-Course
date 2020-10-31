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

tf.app.flags.DEFINE_float('learning_rate', default=0.001, help='initial learning rate')
tf.app.flags.DEFINE_integer('seed', default=111, help='seed')

##################
# Training Flags #
##################
tf.app.flags.DEFINE_integer('batch_size', default=128, help='Batch size for training')
tf.app.flags.DEFINE_integer('num_epoch', default=10, help='Number of training iterations')
tf.app.flags.DEFINE_integer('batch_per_log', default=10, help='Print the log at what number of batches?')

###############
# Model Flags #
###############
tf.app.flags.DEFINE_integer('hidden_size', default=128, help='Number of neurons for RNN hodden layer')

# Store all elemnts in FLAG structure!
args = tf.app.flags.FLAGS


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
