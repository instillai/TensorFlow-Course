=======================
Multi-Class Kernel SVM
=======================

This document is dedicated to explain how to run the python script for this tutorial. For this tutorial, we will create a Kernel SVM for separation of the data. The data that is used for this code is MNIST dataset. This document is inspired on `Implementing Multiclass SVMs <Multiclasssvm_>`_ open source code. However, in ours, we extend it to MNIST dataset and modify its method. 

.. _Multiclasssvm: https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines/06_Implementing_Multiclass_SVMs


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
    
    python multiclass_SVM.py

----------------------------
How to run the code in IDEs?
----------------------------

Since the code is ready-to-go, as long as the TensorFlow can be called in the IDE editor(Pycharm, Spyder,..), the code can be executed successfully.
