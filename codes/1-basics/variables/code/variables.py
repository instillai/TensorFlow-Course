## This code create some arbitrary variables and initialize them ###
# The goal is to show how to define and initialize variables from scratch.

import tensorflow as tf
from tensorflow.python.framework import ops

#######################################
######## Defining Variables ###########
#######################################

# Create three variables with some default values.
weights = tf.Variable(tf.random_normal([2, 3], stddev=0.1),
                      name="weights")
biases = tf.Variable(tf.zeros([3]), name="biases")
custom_variable = tf.Variable(tf.zeros([3]), name="custom")

# Get all the variables' tensors and store them in a list.
all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)


############################################
######## Customized initializer ############
############################################

## Initialation of some custom variables.
## In this part we choose some variables and only initialize them rather than initializing all variables.

# "variable_list_custom" is the list of variables that we want to initialize.
variable_list_custom = [weights, custom_variable]

# The initializer
init_custom_op = tf.variables_initializer(var_list=variable_list_custom )


########################################
######## Global initializer ############
########################################

# Method-1
# Add an op to initialize the variables.
init_all_op = tf.global_variables_initializer()

# Method-2
init_all_op = tf.variables_initializer(var_list=all_variables_list)



##########################################################
######## Initialization using other variables ############
##########################################################

# Create another variable with the same value as 'weights'.
WeightsNew = tf.Variable(weights.initialized_value(), name="WeightsNew")

# Now, the variable must be initialized.
init_WeightsNew_op = tf.variables_initializer(var_list=[WeightsNew])

######################################
####### Running the session ##########
######################################
with tf.Session() as sess:
    # Run the initializer operation.
    sess.run(init_all_op)
    sess.run(init_custom_op)
    sess.run(init_WeightsNew_op)
