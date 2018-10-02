#####################################
# With some tiny modification, this code is the one used by Tensorflow slim at:
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
# Please refer to the link for further explanations.

### The difference is this architecture is written in fully-convolutional fashion.
### The advantage is that, this model can be used for larger image sizes with some average pooling in the last layer.

import tensorflow as tf

slim = tf.contrib.slim


def net_architecture(images, num_classes=10, is_training=False,
                     dropout_keep_prob=0.5,
                     spatial_squeeze=True,
                     scope='Net'):
    """Creates a variant of the Net model.

    Args:
      images: The batch of `Tensors`: size [batch_size, height, width, channels].
      num_classes: Total number of classes.
      is_training: Training/Validation.
      dropout_keep_prob: The percentage of activation values: Only active in training mode!
      scope: Variable_scope.

    Returns:
      logits: the pre-softmax activations of size [batch_size, `num_classes`]
      end_points: The dictionary for the layers outputs.
    """

    # Create empty dictionary
    end_points = {}

    with tf.variable_scope(scope, 'Net', [images, num_classes]) as sc:
        end_points_collection = sc.name + '_end_points'

        # Collect outputs for conv2d and max_pool2d.
        with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.max_pool2d],
                                            outputs_collections=end_points_collection):
            # Layer-1
            net = tf.contrib.layers.conv2d(images, 32, [5, 5], scope='conv1')
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

            # Squeeze spatially to eliminate extra dimensions.(embedding layer)
            if spatial_squeeze:
                logits = tf.squeeze(logits, [1, 2], name='fc4/squeezed')
                end_points[sc.name + '/fc4'] = logits
            return logits, end_points


def net_arg_scope(weight_decay=0.0005, is_training=False):
    """Defines the default network argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the model.
    """
    if is_training:
        with tf.contrib.framework.arg_scope(
                [tf.contrib.layers.conv2d],
                padding='SAME',
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                                   uniform=False, seed=None,
                                                                                   dtype=tf.float32),
                activation_fn=tf.nn.relu) as sc:
            return sc

    else:
        with tf.contrib.framework.arg_scope(
                [tf.contrib.layers.conv2d],
                padding='SAME',
                activation_fn=tf.nn.relu) as sc:
            return sc

