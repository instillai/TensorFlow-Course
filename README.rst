
.. figure:: _img/mainpage/teaser.gif
  :target: https://machinelearningmindset.com/blog/
  
.. figure:: _img/mainpage/subscribe.gif
  :target: https://machinelearningmindset.com/subscription/

********************
`TensorFlow Course`_
********************
.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :target: https://github.com/open-source-for-science/TensorFlow-Course/pulls
.. image:: https://badges.frapsoft.com/os/v2/open-source.svg?v=102
    :target: https://github.com/ellerbrock/open-source-badge/
.. image:: https://img.shields.io/twitter/follow/machinemindset.svg?label=Follow&style=social
    :target: https://twitter.com/machinemindset


This repository aims to provide simple and ready-to-use tutorials for TensorFlow.
Each tutorial includes ``source code`` and most of them are associated with a ``documentation``.

.. .. image:: _img/mainpage/TensorFlow_World.gif

.. The links.
.. _TensorFlow: https://www.tensorflow.org/install/
.. _Wikipedia: https://en.wikipedia.org/wiki/TensorFlow/

#################
Table of Contents
#################
.. contents::
  :local:
  :depth: 3

~~~~~~~~~~~~~~~~~~~~~
What is TensorFlow?
~~~~~~~~~~~~~~~~~~~~~
TensorFlow is an open-source software library for dataflow programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks. It is used for both research and production at Google often replacing its closed-source predecessor, DistBelief.

TensorFlow was developed by the Google Brain team for internal Google use. It was released under the Apache 2.0 open source license on November 9, 2015.

The current stable release as of September 27, 2018 is 1.11.0

============
Motivation
============

There are different motivations for this open source project. TensorFlow (as we write this document) is one of / the best deep learning frameworks available. The question that should be asked is why has this repository been created when there are so many other tutorials about TensorFlow available on the web?

~~~~~~~~~~~~~~~~~~~~~
Why use TensorFlow?
~~~~~~~~~~~~~~~~~~~~~

Deep Learning is in very high interest these days - there's a crucial need for rapid and optimized implementations of the algorithms and architectures. TensorFlow is designed to facilitate this goal.

The strong advantage of TensorFlow is it flexibility in designing highly modular models which can also be a disadvantage for beginners since a lot of the pieces must be considered together when creating the model.

This issue has been facilitated as well by developing high-level APIs such as `Keras <https://keras.io/>`_ and `Slim <https://github.com/tensorflow/models/blob/031a5a4ab41170d555bc3e8f8545cf9c8e3f1b28/research/inception/inception/slim/README.md>`_ which abstract a lot of the pieces used in designing machine learning algorithms.

The interesting thing about TensorFlow is that **it can be found anywhere these days**. Lots of the researchers and developers are using it and *its community is growing at the speed of light*! So many issues can be dealt with easily since they're usually the same issues that a lot of other people run into considering the large number of people involved in the TensorFlow community.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
What's the point of this repository?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Developing open source projects for the sake of just developing something is not the reason behind this effort**.
Considering the large number of tutorials that are being added to this large community, this repository has been created to break the jump-in and jump-out process that usually happens to most of the open source projects, **but why and how**?

First of all, what's the point of putting effort into something that most of the people won't stop by and take a look? What's the point of creating something that does not help anyone in the developers and researchers community? Why spend time for something that can easily be forgotten? But **how we try to do it?** Even up to this
very moment there are countless tutorials on TensorFlow whether on the model design or TensorFlow
workflow.

Most of them are too complicated or suffer from a lack of documentation. There are only a few available tutorials which are concise and well-structured and provide enough insight for their specific implemented models.

The goal of this project is to help the community with structured tutorials and simple and optimized code implementations to provide better insight about how to use TensorFlow *quick and effectively*.

It is worth noting that, **the main goal of this project is to provide well-documented tutorials and less-complicated code**!

=================================================
TensorFlow Installation and Setup the Environment
=================================================

.. image:: _img/mainpage/installation-logo.gif
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: alternate text
   :align: right
   :target: docs/tutorials/installation

.. _TensorFlow Installation: docs/tutorials/installation

In order to install TensorFlow please refer to the following link:

  * `TensorFlow Installation`_


.. image:: _img/mainpage/installation.gif
    :target: https://www.youtube.com/watch?v=_3JFEPk4qQY&t=2s

The virtual environment installation is recommended in order to prevent package conflict and having the capacity to customize the working environment.

====================
TensorFlow Tutorials
====================

The tutorials in this repository are partitioned into relevant categories.

==========================

~~~~~~~~
Warm-up
~~~~~~~~

.. image:: _img/mainpage/welcome.gif
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: alternate text
   :align: right

+----+---------------------+----------------------------------------------------------------------------------------+----------------------------------------------+
| #  |       topic         |   Source Code                                                                          |                                              |
+====+=====================+========================================================================================+==============================================+
| 1  | Start-up            | `Welcome <welcomesourcecode_>`_  / `IPython <ipythonwelcome_>`_                        |  `Documentation <Documentationcnnwelcome_>`_ |
+----+---------------------+----------------------------------------------------------------------------------------+----------------------------------------------+

==========================

~~~~~~
Basics
~~~~~~

.. image:: _img/mainpage/basics.gif
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: alternate text
   :align: right

+----+---------------------+----------------------------------------------------------------------------------------+----------------------------------------------+
| #  |       topic         |   Source Code                                                                          |                                              |
+====+=====================+========================================================================================+==============================================+
| 2  | *TensorFLow Basics* | `Basic Math Operations <basicmathsourcecode_>`_   / `IPython <ipythonbasicmath_>`_     |  `Documentation <Documentationbasicmath_>`_  |
+----+---------------------+----------------------------------------------------------------------------------------+----------------------------------------------+
| 3  | *TensorFLow Basics* | `TensorFlow Variables <variablssourcecode_>`_   / `IPython <ipythonvariabls_>`_        |  `Documentation <Documentationvariabls_>`_   |
+----+---------------------+----------------------------------------------------------------------------------------+----------------------------------------------+

==========================

~~~~~~~~~~~~~~~~~~~~~~
Basic Machine Learning
~~~~~~~~~~~~~~~~~~~~~~

.. image:: _img/mainpage/basicmodels.gif
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: alternate text
   :align: right

+----+----------------------------+----------------------------------------------------------------------------------------+----------------------------------------------+
| #  |       topic                |   Source Code                                                                          |                                              |
+====+============================+========================================================================================+==============================================+
| 4  | *Linear Models*            |`Linear Regression`_  / `IPython <LinearRegressionipython_>`_                           | `Documentation <Documentationlr_>`_          |
+----+----------------------------+----------------------------------------------------------------------------------------+----------------------------------------------+
| 5  | *Predictive Models*        | `Logistic Regression`_  / `IPython <LogisticRegressionipython_>`_                      | `Documentation <LogisticRegDOC_>`_           |
+----+----------------------------+----------------------------------------------------------------------------------------+----------------------------------------------+
| 6  | *Support Vector Machines*  | `Linear SVM`_  / `IPython <LinearSVMipython_>`_                                        |                                              |
+----+----------------------------+----------------------------------------------------------------------------------------+----------------------------------------------+
| 7  | *Support Vector Machines*  |`MultiClass Kernel SVM`_  / `IPython <MultiClassKernelSVMipython_>`_                    |                                              |
+----+----------------------------+----------------------------------------------------------------------------------------+----------------------------------------------+

==========================

~~~~~~~~~~~~~~~~
Neural Networks
~~~~~~~~~~~~~~~~

.. image:: _img/mainpage/CNNs.png
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: alternate text
   :align: right

+----+-----------------------------------+-----------------------------------------------------------------------------------------------+----------------------------------------------+
| #  |       topic                       |   Source Code                                                                                 |                                              |
+====+===================================+===============================================================================================+==============================================+
| 8  | *Multi Layer Perceptron*          |`Simple Multi Layer Perceptron`_   / `IPython <MultiLayerPerceptronipython_>`_                 |                                              |
+----+-----------------------------------+-----------------------------------------------------------------------------------------------+----------------------------------------------+
| 9  | *Convolutional Neural Network*    | `Simple Convolutional Neural Networks`_                                                       |       `Documentation <Documentationcnn_>`_   |
+----+-----------------------------------+-----------------------------------------------------------------------------------------------+----------------------------------------------+
| 10 | *Recurrent Neural Network*        | `RNN`_  / `IPython <RNNIpython_>`_                                                            |                                              |
+----+-----------------------------------+-----------------------------------------------------------------------------------------------+----------------------------------------------+

.. ~~~~~~~~~~~~
.. **Welcome**
.. ~~~~~~~~~~~~

.. The tutorial in this section is just a simple entrance to TensorFlow.

.. _welcomesourcecode: codes/python/0-welcome
.. _Documentationcnnwelcome: docs/tutorials/0-welcome
.. _ipythonwelcome: codes/ipython/0-welcome/code/0-welcome.ipynb



.. +---+---------------------------------------------+-------------------------------------------------+
.. | # |          Source Code                        |                                                 |
.. +===+=============================================+=================================================+
.. | 1 |    `Welcome <welcomesourcecode_>`_          |  `Documentation <Documentationcnnwelcome_>`_    |
.. +---+---------------------------------------------+-------------------------------------------------+

.. ~~~~~~~~~~
.. **Basics**
.. ~~~~~~~~~~
.. These tutorials are related to basics of TensorFlow.

.. _basicmathsourcecode: codes/python/1-basics/basic_math_operations
.. _Documentationbasicmath: docs/tutorials/1-basics/basic_math_operations
.. _ipythonbasicmath: codes/ipython/1-basics/basic_math_operations/code/basic_math_operation.ipynb

.. _ipythonvariabls: codes/ipython/1-basics/variables/code/variables.ipynb
.. _variablssourcecode: codes/python/1-basics/variables/README.rst
.. _Documentationvariabls: docs/tutorials/1-basics/variables


.. +---+-----------------------------------------------------+-------------------------------------------------+
.. | # |          Source Code                                |                                                 |
.. +===+=====================================================+=================================================+
.. | 1 |    `Basic Math Operations <basicmathsourcecode_>`_  |  `Documentation <Documentationbasicmath_>`_     |
.. +---+-----------------------------------------------------+-------------------------------------------------+
.. | 2 |    `TensorFlow Variables <variablssourcecode_>`_    |  `Documentation <Documentationvariabls_>`_      |
.. +---+-----------------------------------------------------+-------------------------------------------------+

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. **Machine Learning Basics**
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. We are going to present concepts of basic machine learning models and methods and show how to implement them in Tensorflow.

.. _Linear Regression: codes/python/2-basics_in_machine_learning/linear_regression
.. _LinearRegressionipython: codes/ipython/2-basics_in_machine_learning/linear_regression/code/linear_regression.ipynb
.. _Documentationlr: docs/tutorials/2-basics_in_machine_learning/linear_regression

.. _Logistic Regression: codes/python/2-basics_in_machine_learning/logistic_regression
.. _LogisticRegressionipython: codes//ipython/2-basics_in_machine_learning/logistic_regression/code/logistic_regression.ipynb
.. _LogisticRegDOC: docs/tutorials/2-basics_in_machine_learning/logistic_regression

.. _Linear SVM: codes/python/2-basics_in_machine_learning/linear_svm
.. _LinearSVMipython: codes//ipython/2-basics_in_machine_learning/linear_svm/code/linear_svm.ipynb


.. _MultiClass Kernel SVM: codes/python/2-basics_in_machine_learning/multiclass_svm
.. _MultiClassKernelSVMipython: codes/ipython/2-basics_in_machine_learning/multiclass_svm/code/multiclass_svm.ipynb


.. +---+---------------------------------------------+----------------------------------------+
.. | # |          Source Code                        |                                        |
.. +===+=============================================+========================================+
.. | 1 |    `Linear Regression`_                     |  `Documentation <Documentationlr_>`_   |
.. +---+---------------------------------------------+----------------------------------------+
.. | 2 |    `Logistic Regression`_                   |  `Documentation <LogisticRegDOC_>`_    |
.. +---+---------------------------------------------+----------------------------------------+
.. | 3 |    `Linear SVM`_                            |                                        |
.. +---+---------------------------------------------+----------------------------------------+
.. | 4 |    `MultiClass Kernel SVM`_                 |                                        |
.. +---+---------------------------------------------+----------------------------------------+

.. ~~~~~~~~~~~~~~~~~~~
.. **Neural Networks**
.. ~~~~~~~~~~~~~~~~~~~
.. The tutorials in this section are related to neural network architectures.

.. _Simple Convolutional Neural Networks: codes/python/3-neural_networks/convolutional-neural-network
.. _Documentationcnn: docs/tutorials/3-neural_network/convolutiona_neural_network

.. _Simple Multi Layer Perceptron: codes/python/3-neural_networks/multi-layer-perceptron
.. _MultiLayerPerceptronipython: codes/ipython/3-neural_networks/multi-layer-perceptron/code/train_mlp.ipynb

.. _RNN: codes/python/3-neural_networks/recurrent-neural-networks/code/rnn.py
.. _RNNIpython: codes/ipython/3-neural_networks/recurrent-neural-networks/code/rnn.ipynb


.. +---+---------------------------------------------+----------------------------------------+
.. | # |          Source Code                        |                                        |
.. +===+=============================================+========================================+
.. | 1 |    `Multi Layer Perceptron`_                |                                        |
.. +---+---------------------------------------------+----------------------------------------+
.. | 2 |    `Convolutional Neural Networks`_         |  `Documentation <Documentationcnn_>`_  |
.. +---+---------------------------------------------+----------------------------------------+


=====================
Some Useful Tutorials
=====================

  * `TensorFlow Examples <https://github.com/aymericdamien/TensorFlow-Examples>`_ - TensorFlow tutorials and code examples for beginners
  * `Sungjoon's TensorFlow-101 <https://github.com/sjchoi86/Tensorflow-101>`_ - TensorFlow tutorials written in Python with Jupyter Notebook
  * `Terry Um’s TensorFlow Exercises <https://github.com/terryum/TensorFlow_Exercises>`_ - Re-create the codes from other TensorFlow examples
  * `Classification on time series <https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition>`_ - Recurrent Neural Network classification in TensorFlow with LSTM on cellphone sensor data

=============
Contributing
=============

When contributing to this repository, please first discuss the change you wish to make via issue,
email, or any other method with the owners of this repository before making a change. *For typos, please
do not create a pull request. Instead, declare them in issues or email the repository owner*.

Please note we have a code of conduct, please follow it in all your interactions with the project.

~~~~~~~~~~~~~~~~~~~~
Pull Request Process
~~~~~~~~~~~~~~~~~~~~

Please consider the following criterions in order to help us in a better way:

  * The pull request is mainly expected to be a code script suggestion or improvement.
  * A pull request related to non-code-script sections is expected to make a significant difference in the documentation. Otherwise, it is expected to be announced in the issues section.
  * Ensure any install or build dependencies are removed before the end of the layer when doing a build and creating a pull request.
  * Add comments with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.
  * You may merge the Pull Request in once you have the sign-off of at least one other developer, or if you do not have permission to do that, you may request the owner to merge it for you if you believe all checks are passed.


~~~~~~~~~~~
Final Note
~~~~~~~~~~~

We are looking forward to your kind feedback. Please help us to improve this open source project and make our work better.
For contribution, please create a pull request and we will investigate it promptly. Once again, we appreciate
your kind feedback and elaborate code inspections.

================
Acknowledgement
================

I have taken huge efforts in this project for hopefully being a small part of TensorFlow world. However, it would not have been plausible without the kind support and help of my friend and colleague `Domenick Poster <https://github.com/vonclites/>`_ for his valuable advices. He helped me for having a better understanding of TensorFlow and my special appreciation goes to him.
