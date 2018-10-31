
===========================
Welcome to TensorFlow World
===========================

This document is dedicated to explain how to run the python script for this tutorial.

---------------------------
Test TensorFlow Environment
---------------------------

``WARNING:`` If TensorFlow is installed in any environment(virtual environment, ...), it must be activated at first. So at first make sure the tensorFlow is available in the current environment using the following script:

.. code:: shell

    cd code/
    python TensorFlow_Test.py
    
--------------------------------
How to run the code in Terminal?
--------------------------------

    
Please root to the ``code/`` directory and run the python script as the general form of below:

.. code:: shell
    
    python [python_code_file.py] --log_dir='absolute/path/to/log_dir'
    

As an example the code can be executed as follows:

.. code:: shell
    
    python 1-welcome.py --log_dir='~/log_dir'

The ``--log_dir`` flag is to provide the address which the event files (for visualizing in Tensorboard) will be saved. The flag of ``--log_dir`` is not required because its default value is available in the source code as follows:

.. code:: python
    
    tf.app.flags.DEFINE_string(
    'log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
    'Directory where event logs are written to.')

----------------------------
How to run the code in IDEs?
----------------------------

Since the code is ready-to-go, as long as the TensorFlow can be called in the IDE editor(Pycharm, Spyder,..), the code can be executed successfully.


----------------------------
How to run the Tensorboard?
----------------------------
.. _Google’s words: https://www.tensorflow.org/get_started/summaries_and_tensorboard
TensorBoard is the graph visualization tools provided by TensorFlow. Using `Google’s words`_: “The computations you'll use TensorFlow for - like training a massive deep neural network - can be complex and confusing. To make it easier to understand,
debug, and optimize TensorFlow programs, we've included a suite of visualization tools called
TensorBoard.”

The Tensorboard can be run as follows in the terminal:

.. code:: shell
    
    tensorboard --logdir="absolute/path/to/log_dir"


 



