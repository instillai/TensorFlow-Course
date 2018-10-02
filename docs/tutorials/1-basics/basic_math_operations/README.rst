============================
Welcome to TensorFlow World
============================

.. _this link: https://github.com/astorfi/TensorFlow-World/tree/master/codes/0-welcome

The tutorials in this section are just a start for going into the TensorFlow world.

We using Tensorboard for visualizing the outcomes. TensorBoard is the graph visualization tools provided by TensorFlow. Using Google’s words: “The computations you'll use TensorFlow for - like training a massive deep neural network - can be complex and confusing. To make it easier to understand, debug, and optimize TensorFlow programs, we've included a suite of visualization tools called TensorBoard.” A simple Tensorboard implementation is used in this tutorial. 

**NOTE:*** 
     
     * The details of summary operations, Tensorboard, and their advantages are beyond the scope of this tutorial and will be presented in more advanced tutorials.


--------------------------
Preparing the environment
--------------------------

At first, we have to import the necessary libraries.

.. code:: python
    
       from __future__ import print_function
       import tensorflow as tf
       import os

Since we are aimed to use Tensorboard, we need a directory to store the information (the operations and their corresponding outputs if desired by the user). This information is exported to ``event files`` by TensorFlow. The even files can be transformed to visual data such that the user is able to evaluate the architecture and the operations. The ``path`` to store these even files is defined as below:

.. code:: python
    
       # The default path for saving event files is the same folder of this python file.
       tf.app.flags.DEFINE_string(
       'log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
       'Directory where event logs are written to.')

       # Store all elements in FLAG structure!
       FLAGS = tf.app.flags.FLAGS

The ``os.path.dirname(os.path.abspath(__file__))`` gets the directory name of the current python file. The ``tf.app.flags.FLAGS`` points to all defined flags using the ``FLAGS`` indicator. From now on the flags can be called using ``FLAGS.flag_name``.

For convenience, it is useful to only work with ``absolute paths``. By using the following script, the user is prompt to use absolute paths for the ``log_dir`` directory.

.. code:: python

    # The user is prompted to input an absolute path.
    # os.path.expanduser is leveraged to transform '~' sign to the corresponding path indicator.
    #       Example: '~/logs' equals to '/home/username/logs'
    if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
        raise ValueError('You must assign absolute path for --log_dir')

--------
Basics
--------

Some basic math operations can be defined by TensorFlow:

.. code:: python

     # Defining some constant values
     a = tf.constant(5.0, name="a")
     b = tf.constant(10.0, name="b")

     # Some basic operations
     x = tf.add(a, b, name="add")
     y = tf.div(a, b, name="divide")
    
The ``tf.`` operator performs the specific operation and the output will be a ``Tensor``. The attribute ``name="some_name"`` is defined for better Tensorboard visualization as we see later in this tutorial.

-------------------
Run the Experiment
-------------------

The ``session``, which is the environment for running the operations, is executed as below:

.. code:: python

    # Run the session
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
        print("output: ", sess.run(welcome))

    # Closing the writer.
    writer.close()
    sess.close()

The ``tf.summary.FileWriter`` is defined to write the summaries into ``event files``.The command of ``sess.run()`` must be used for evaluation of any ``Tensor`` otherwise the operation won't be executed. In the end by using the ``writer.close()``, the summary writer will be closed.
    
--------
Results
--------

The results for running in the terminal is as bellow:

.. code:: shell

        a = 5.0
        b = 10.0
        a + b = 15.0
        a/b = 0.5



If we run the Tensorboard using ``tensorboard --logdir="absolute/path/to/log_dir"`` we get the following when visualiaing the ``Graph``:

.. figure:: https://github.com/astorfi/TensorFlow-World/blob/master/docs/_img/1-basics/basic_math_operations/graph-run.png
   :scale: 30 %
   :align: center

   **Figure 1:** The TensorFlow Graph.

