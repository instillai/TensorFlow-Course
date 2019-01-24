Introduction to TensorFlow Variables: Creation, Initialization
--------------------------------------------------------------

This tutorial deals with defining and initializing TensorFlow variables.

Introduction
------------

Defining ``variables`` is necessary because they hold the parameters.
Without having parameters, training, updating, saving, restoring and any
other operations cannot be performed. The defined variables in
TensorFlow are just tensors with certain shapes and types. The tensors
must be initialized with values to become valid. In this tutorial, we
are going to explain how to ``define`` and ``initialize`` variables. The
`source
code <https://github.com/astorfi/TensorFlow-World/tree/master/codes/1-basics/variables>`__
is available on the dedicated GitHub repository.

Creating variables
------------------

For a variable generation, the class of tf.Variable() will be used. When
we define a variable, we basically pass a ``tensor`` and its ``value``
to the graph. Basically, the following will happen:

    -  A ``variable`` tensor that holds a value will be pass to the
       graph.
    -  By using tf.assign, an initializer set initial variable value.

Some arbitrary variables can be defined as follows:

.. code:: python

     
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
    

In the above script, ``ops.get_collection`` gets the list of all defined variables
from the defined graph. The "name" key, define a specific name for each
variable on the graph

Initialization
--------------

``Initializers`` of the variables must be run before all other
operations in the model. For an analogy, we can consider the starter of
the car. Instead of running an initializer, variables can be
``restored`` too from saved models such as a checkpoint file. Variables
can be initialized globally, specifically, or from other variables. We
investigate different choices in the subsequent sections.

Initializing Specific Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By using tf.variables\_initializer, we can explicitly command the
TensorFlow to only initialize a certain variable. The script is as follows

.. code:: python
     
    # "variable_list_custom" is the list of variables that we want to initialize.
    variable_list_custom = [weights, custom_variable]

    # The initializer
    init_custom_op = tf.variables_initializer(var_list=variable_list_custom)

Noted that custom initialization does not mean that we don't need to
initialize other variables! All variables that some operations will be
done upon them over the graph, must be initialized or restored from
saved variables. This only allows us to realize how we can initialize
specific variables by hand.

Global variable initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All variables can be initialized at once using the
tf.global\_variables\_initializer(). This op must be run after the model constructed. 
The script is as below:

.. code:: python
     
    # Method-1
    # Add an op to initialize the variables.
    init_all_op = tf.global_variables_initializer()

    # Method-2
    init_all_op = tf.variables_initializer(var_list=all_variables_list)

Both the above methods are identical. We only provide the second one to
demonstrate that the ``tf.global_variables_initializer()`` is nothing
but ``tf.variables_initializer`` when you yield all the variables as the input argument.

Initialization of a variables using other existing variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

New variables can be initialized using other existing variables' initial
values by taking the values using initialized\_value().

Initialization using predefined variables' values

.. code:: python

    # Create another variable with the same value as 'weights'.
    WeightsNew = tf.Variable(weights.initialized_value(), name="WeightsNew")

    # Now, the variable must be initialized.
    init_WeightsNew_op = tf.variables_initializer(var_list=[WeightsNew])

As it can be seen from the above script, the ``WeightsNew`` variable is
initialized with the values of the ``weights`` predefined value.

Running the session
-------------------

All we did so far was to define the initializers' ops and put them on the
graph. In order to truly initialize variables, the defined initializers'
ops must be run in the session. The script is as follows:

Running the session for initialization

.. code:: python

    with tf.Session() as sess:
        # Run the initializer operation.
        sess.run(init_all_op)
        sess.run(init_custom_op)
        sess.run(init_WeightsNew_op)

Each of the initializers has been run separated using a session.

Summary
-------

In this tutorial, we walked through the variable creation and
initialization. The global, custom and inherited variable initialization
have been investigated. In the future posts, we investigate how to save
and restore the variables. Restoring a variable eliminate the necessity
of its initialization.

