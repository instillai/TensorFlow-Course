==================
Linear Regression
==================

This document is dedicated to explain how to run the python script for this tutorial. The documentation is available `here <Documentationlinearregression_>`_. Alternatively, you can check this ``Linear Regression using TensorFlow`` `blog post <blogpostlinearregression_>`_ for further details.

.. _blogpostlinearregression: http://www.machinelearninguru.com/deep_learning/tensorflow/machine_learning_basics/linear_regresstion/linear_regression.html

.. _Documentationlinearregression: https://github.com/astorfi/TensorFlow-World/wiki/Linear-Regeression

-------------------
Python Environment
-------------------

``WARNING:`` If TensorFlow is installed in any environment(virtual environment, ...), it must be activated at first. So at first make sure the tensorFlow is available in the current environment using the following script:

--------------------------------
How to run the code in Terminal?
--------------------------------

    
Please root to the ``code/`` directory and run the python script as the general form of below:

.. code:: shell
    
    python [python_code_file.py] 
    

As an example the code can be executed as follows:

.. code:: shell
    
    python linear_regression.py --num_epochs=50

The ``--num_epochs`` flag is to provide the number of epochs that will be used for training. The ``--num_epochs`` flag is not required because its default value is ``50`` and is provided in the source code as follows:

.. code:: python
    
    tf.app.flags.DEFINE_integer(
    'num_epochs', 50, 'The number of epochs for training the model. Default=50')

----------------------------
How to run the code in IDEs?
----------------------------

Since the code is ready-to-go, as long as the TensorFlow can be called in the IDE editor(Pycharm, Spyder,..), the code can be executed successfully.
