==================
Logistic Regression
==================

This document is dedicated to explaining how to run the python script for this tutorial. ``Logistic regression`` is a binary
classification algorithm in which `yes` or `no` are the only possible responses. The linear output is transformed to a probability of course between zero and 1. The decision is made by thresholding the probability and saying it belongs to which class. We consider ``Softmax`` with ``cross entropy`` loss for minimizing the loss.

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
    
    python logistic_regression.py --num_epochs=50 --batch_size=512 --max_num_checkpoint=10 --num_classes=2

Different ``flags`` are provided for training. For the full list please refer to the source code. The above example is just an example as is!

----------------------------
How to run the code in IDEs?
----------------------------

Since the code is ready-to-go, as long as the TensorFlow can be called in the IDE editor(Pycharm, Spyder,..), the code can be executed successfully.
