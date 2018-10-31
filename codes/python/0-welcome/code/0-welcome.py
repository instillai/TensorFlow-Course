#####################################################
########## Welcome to TensorFlow World ##############
#####################################################

# The tutorials in this section is just a start for going into TensorFlow world.
# The TensorFlow flags are used for having a more user friendly environment.

from __future__ import print_function
import tensorflow as tf
import os


######################################
######### Necessary Flags ############
# ####################################

log_dir = os.path.dirname(os.path.abspath(__file__)) + '/logs'

################################################
################# handling errors!##############
################################################

# Defining some sentence!
welcome = tf.constant('Welcome to TensorFlow world!')

# Run the session
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.expanduser(log_dir), sess.graph)
    print("output: ", sess.run(welcome))

# Closing the writer.
writer.close()
sess.close()


