

********************
`TensorFlow Course`_
********************
.. image:: https://travis-ci.org/instillai/TensorFlow-Course.svg?branch=master
    :target: https://travis-ci.org/instillai/TensorFlow-Course
.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :target: https://github.com/open-source-for-science/TensorFlow-Course/pulls
.. image:: https://img.shields.io/twitter/follow/machinemindset.svg?label=Follow&style=social
    :target: https://twitter.com/machinemindset
.. image:: https://zenodo.org/badge/151300862.svg
   :target: https://zenodo.org/badge/latestdoi/151300862


This repository aims to provide simple and ready-to-use tutorials for TensorFlow.
Each tutorial includes ``source code`` and most of them are associated with a ``documentation``.

.. .. image:: _img/mainpage/TensorFlow_World.gif

.. The links.
.. _TensorFlow: https://www.tensorflow.org/install/
.. _Wikipedia: https://en.wikipedia.org/wiki/TensorFlow/


##########################################################################
Sponsorship
##########################################################################

To support maintaining and upgrading this project, please kindly consider `Sponsoring the project developer <https://github.com/sponsors/astorfi/dashboard>`_.

Any level of support is a great contribution here :heart:

**Status:** *This project has been updated to **TensorFlow 2.3**.*


#################
Table of Contents
#################
.. contents::
  :local:
  :depth: 3


==========================================
Download Free TensorFlow Roadmap EBook
==========================================

.. raw:: html

   <div align="center">

.. raw:: html

 <a href="http://www.machinelearningmindset.com/tensorflow-roadmap-ebook/" target="_blank">
  <img width="710" height="500" align="center" src="https://github.com/machinelearningmindset/TensorFlow-Course/blob/master/_img/mainpage/booksubscribe.png"/>
 </a>

.. raw:: html

   </div>

==========================================
Slack Group
==========================================

.. raw:: html

   <div align="center">

.. raw:: html

 <a href="https://www.machinelearningmindset.com/slack-group/" target="_blank">
  <img width="1033" height="350" align="center" src="https://github.com/machinelearningmindset/TensorFlow-Course/blob/master/_img/0-welcome/joinslack.png"/>
 </a>

.. raw:: html

   </div>



~~~~~~~~~~~~~~~~~~~~~
What is TensorFlow?
~~~~~~~~~~~~~~~~~~~~~
TensorFlow is an open-source software library for dataflow programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks. It is used for both research and production at Google often replacing its closed-source predecessor, DistBelief.

TensorFlow was developed by the Google Brain team for internal Google use. It was released under the Apache 2.0 open source license on November 9, 2015.


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

.. _TensorFlow Installation: https://www.tensorflow.org/install

In order to install TensorFlow please refer to the following link:

  * `TensorFlow Installation`_


.. image:: _img/mainpage/installation.gif
    :target: https://www.tensorflow.org/install

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


.. _colab: https://colab.research.google.com/github/instillai/TensorFlow-Course/blob/master/codes/ipython/0-welcome/welcome.ipynb
.. _Documentationcnnwelcome: docs/tutorials/0-welcome
.. _ipythonwelcome: codes/ipython/0-welcome/welcome.ipynb
.. _pythonwelcome: https://github.com/instillai/TensorFlow-Course/blob/master/codes/python/0-welcome/welcome.py
.. _videowelcome: https://youtu.be/xd0DVygHlNE


.. |Welcome| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/instillai/TensorFlow-Course/blob/master/codes/ipython/0-welcome/welcome.ipynb

.. |youtubeim| image:: _img/mainpage/YouTube.png
  :target: https://github.com/instillai/TensorFlow-Course/blob/master/_img/mainpage/YouTube.png


+----+---------------------+--------------------------+------------------------------------------------------------------------+-------------------------------------------+
| #  |       topic         |          Run             |  Source Code                                                           |  Media                                    |
+====+=====================+==========================+========================================================================+===========================================+
| 1  | Start-up            |       |Welcome|          | `Notebook <ipythonwelcome_>`_  / `Python <pythonwelcome_>`_            | `Video Tutorial <videowelcome_>`_         |
+----+---------------------+--------------------------+------------------------------------------------------------------------+-------------------------------------------+

==========================

~~~~~~
Basics
~~~~~~

.. raw:: html

   <div align="left">

.. raw:: html

 <a href="https://github.com/instillai/TensorFlow-Course/blob/master/_img/mainpage/basics.gif" target="_blank">
  <img width="250" height="250" align="center" src="https://github.com/instillai/TensorFlow-Course/blob/master/_img/mainpage/basics.gif"/>
 </a>

.. raw:: html

   </div>

.. raw:: html

   <br>



.. _ipythontensors: codes/ipython/1-basics/tensors.ipynb
.. _pythontensors: codes/python/1-basics/tensors.py
.. _videotensors: https://youtu.be/Od-VvnYUbFw
.. |Tensors| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/instillai/TensorFlow-Course/blob/master/codes/ipython/1-basics/tensors.ipynb

.. _ipythonad: codes/ipython/1-basics/automatic_differentiation.ipynb
.. _pythonad: codes/python/1-basics/automatic_differentiation.py
.. _videoad: https://youtu.be/l-MGydWW-UE
.. |AD| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/instillai/TensorFlow-Course/blob/master/codes/ipython/1-basics/automatic_differentiation.ipynb

.. _ipythongraphs: codes/ipython/1-basics/graph.ipynb
.. _pythongraphs: codes/python/1-basics/graph.py
.. _videographs: https://youtu.be/P9xA1s6AUNk
.. |graphs| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/instillai/TensorFlow-Course/blob/master/codes/ipython/1-basics/graph.ipynb


.. _ipythonmodels: codes/ipython/1-basics/models.ipynb
.. _pythonmodels: codes/python/1-basics/models.py
.. _videomodels: https://youtu.be/WnlUE04REOY
.. |models| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/instillai/TensorFlow-Course/blob/master/codes/ipython/1-basics/models.ipynb



+----+-----------------------------------+--------------------------+------------------------------------------------------------------------+-----------------------------------------+
| #  |       topic                       |          Run             |  Source Code                                                           |        Media                            |
+====+===================================+==========================+========================================================================+=========================================+
| 1  | Tensors                           |       |Tensors|          | `Notebook <ipythontensors_>`_  / `Python <pythontensors_>`_            | `Video Tutorial <videotensors_>`_       |
+----+-----------------------------------+--------------------------+------------------------------------------------------------------------+-----------------------------------------+
| 2  | Automatic Differentiation         |       |AD|               | `Notebook <ipythonad_>`_  / `Python <pythonad_>`_                      | `Video Tutorial <videoad_>`_            |
+----+-----------------------------------+--------------------------+------------------------------------------------------------------------+-----------------------------------------+
| 3  | Introduction to Graphs            |       |graphs|           | `Notebook <ipythongraphs_>`_ / `Python <pythongraphs_>`_               | `Video Tutorial <videographs_>`_        |
+----+-----------------------------------+--------------------------+------------------------------------------------------------------------+-----------------------------------------+
| 4  | TensorFlow Models                 |       |models|           | `Notebook <ipythonmodels_>`_  / `Python <pythonmodels_>`_              | `Video Tutorial <videomodels_>`_        |
+----+-----------------------------------+--------------------------+------------------------------------------------------------------------+-----------------------------------------+

==========================

~~~~~~~~~~~~~~~~~~~~~~
Basic Machine Learning
~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <div align="left">

.. raw:: html

 <a href="https://github.com/instillai/TensorFlow-Course/blob/master/_img/mainpage/basicmodels.gif" target="_blank">
  <img width="250" height="250" align="center" src="https://github.com/instillai/TensorFlow-Course/blob/master/_img/mainpage/basicmodels.gif"/>
 </a>

.. raw:: html

   </div>

.. raw:: html

   <br>

.. .. image:: _img/mainpage/basicmodels.gif
..    :height: 100px
..    :width: 200 px
..    :scale: 50 %
..    :alt: alternate text
..    :align: right


.. _ipythonlinearreg: codes/ipython/basics_in_machine_learning/linearregression.ipynb
.. _pythonlinearreg: codes/python/basics_in_machine_learning/linearregression.py
.. _tutoriallinearreg: https://www.machinelearningmindset.com/linear-regression-with-tensorflow/
.. _videoinearreg: https://youtu.be/2RTBBiKKuLI

.. _tutorialdataaugmentation: https://www.machinelearningmindset.com/data-augmentation-with-tensorflow/
.. _ipythondataaugmentation: https://github.com/instillai/TensorFlow-Course/blob/master/codes/ipython/basics_in_machine_learning/dataaugmentation.ipynb
.. _pythondataaugmentation: https://github.com/instillai/TensorFlow-Course/blob/master/codes/python/basics_in_machine_learning/dataaugmentation.py
.. _videodataaugmentation: https://youtu.be/HbzR2snHJF0

.. |lr| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/instillai/TensorFlow-Course/blob/master/codes/ipython/basics_in_machine_learning/linearregression.ipynb
.. |da| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/instillai/TensorFlow-Course/blob/master/codes/ipython/basics_in_machine_learning/dataaugmentation.ipynb


+----+-----------------------------------+--------------------------+------------------------------------------------------------------------------------+----------------------------------------------+----------------------------------------------+
| #  |       topic                       |          Run             |  Source Code                                                                       |  More                                        |           Media                              |
+====+===================================+==========================+====================================================================================+==============================================+==============================================+
| 1  | Linear Regression                 |       |lr|               | `Notebook <ipythonlinearreg_>`_  / `Python <pythonlinearreg_>`_                    | `Tutorial <tutoriallinearreg_>`_             | `Video Tutorial <videoinearreg_>`_           |
+----+-----------------------------------+--------------------------+------------------------------------------------------------------------------------+----------------------------------------------+----------------------------------------------+
| 2  | Data Augmentation                 |       |da|               | `Notebook <ipythondataaugmentation_>`_ / `Python <pythondataaugmentation_>`_       | `Tutorial <tutorialdataaugmentation_>`_      | `Video Tutorial <videodataaugmentation_>`_   |
+----+-----------------------------------+--------------------------+------------------------------------------------------------------------------------+----------------------------------------------+----------------------------------------------+



.. +----+----------------------------+----------------------------------------------------------------------------------------+----------------------------------------------+

==========================

~~~~~~~~~~~~~~~~
Neural Networks
~~~~~~~~~~~~~~~~

.. raw:: html

   <div align="left">

.. raw:: html

 <a href="https://github.com/instillai/TensorFlow-Course/blob/master/_img/mainpage/CNNs.png" target="_blank">
  <img width="600" height="180" align="center" src="https://github.com/instillai/TensorFlow-Course/blob/master/_img/mainpage/CNNs.png"/>
 </a>

.. raw:: html

   </div>

.. raw:: html

    <br>


.. _ipythonmlp: https://github.com/instillai/TensorFlow-Course/blob/master/codes/ipython/neural_networks/mlp.ipynb
.. _pythonmlp: https://github.com/instillai/TensorFlow-Course/blob/master/codes/python/neural_networks/mlp.py
.. _videomlp: https://youtu.be/w20efZqSK2Y

.. _ipythoncnn: https://github.com/instillai/TensorFlow-Course/blob/master/codes/ipython/neural_networks/CNNs.ipynb
.. _pythoncnn: https://github.com/instillai/TensorFlow-Course/blob/master/codes/python/neural_networks/cnns.py
.. _videocnn: https://youtu.be/WVifZBCRz8g


.. |mlp| image:: https://colab.research.google.com/assets/colab-badge.svg
 :target: https://colab.research.google.com/github/instillai/TensorFlow-Course/blob/master/codes/ipython/neural_networks/mlp.ipynb
.. |cnn| image:: https://colab.research.google.com/assets/colab-badge.svg
 :target: https://colab.research.google.com/github/instillai/TensorFlow-Course/blob/master/codes/ipython/neural_networks/CNNs.ipynb


+----+------------------------------------------+--------------------------+------------------------------------------------------+------------------------------------+
| #  |       topic                              |          Run             |  Source Code                                         |            Media                   |
+====+==========================================+==========================+======================================================+====================================+
| 1  |  *Multi Layer Perceptron*                |       |mlp|              | `Notebook <ipythonmlp_>`_ / `Python <pythonmlp_>`_   | `Video Tutorial <videomlp_>`_      |
+----+------------------------------------------+--------------------------+------------------------------------------------------+------------------------------------+
| 2  |  *Convolutional Neural Networks*         |       |cnn|              | `Notebook <ipythoncnn_>`_ / `Python <pythoncnn_>`_   | `Video Tutorial <videocnn_>`_      |
+----+------------------------------------------+--------------------------+------------------------------------------------------+------------------------------------+

==========================

~~~~~~~~~~~~~~~~
Advanced
~~~~~~~~~~~~~~~~


.. raw:: html

   <div align="left">

.. raw:: html

 <a href="https://github.com/instillai/TensorFlow-Course/blob/master/_img/mainpage/Build.png" target="_blank">
  <img width="180" height="180" align="center" src="https://github.com/instillai/TensorFlow-Course/blob/master/_img/mainpage/Build.png"/>
 </a>

.. raw:: html

   </div>

.. raw:: html

    <br>




.. _ipythoncustomtr: https://github.com/instillai/TensorFlow-Course/blob/master/codes/ipython/advanced/custom_training.ipynb
.. _pythoncustomtr: https://github.com/instillai/TensorFlow-Course/blob/master/codes/python/advanced/custom_training.py
.. _videocustomtr: https://youtu.be/z5gcabfyPfA

.. _ipythondgenerator: https://github.com/instillai/TensorFlow-Course/blob/master/codes/ipython/advanced/dataset_generator.ipynb
.. _pythondgenerator: https://github.com/instillai/TensorFlow-Course/blob/master/codes/python/advanced/dataset_generator.py
.. _videodgenerator: https://youtu.be/-YsgMdDPu3g

.. _ipythontfrecords: https://github.com/instillai/TensorFlow-Course/blob/master/codes/ipython/advanced/tfrecords.ipynb
.. _pythontfrecords: https://github.com/instillai/TensorFlow-Course/blob/master/codes/python/advanced/tfrecords.py
.. _videotfrecords: https://youtu.be/zqavy_5QMk8


.. |ctraining| image:: https://colab.research.google.com/assets/colab-badge.svg
 :target: https://colab.research.google.com/github/instillai/TensorFlow-Course/blob/master/codes/ipython/advanced/custom_training.ipynb

.. |dgenerator| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/instillai/TensorFlow-Course/blob/master/codes/ipython/advanced/dataset_generator.ipynb

.. |tfrecords| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/instillai/TensorFlow-Course/blob/master/codes/ipython/advanced/tfrecords.ipynb


+----+------------------------------------------+--------------------------+--------------------------------------------------------------------+----------------------------------------+
| #  |       topic                              |          Run             |  Source Code                                                       |           Media                        |
+====+==========================================+==========================+====================================================================+========================================+
| 1  |  *Custom Training*                       |       |ctraining|        | `Notebook <ipythoncustomtr_>`_ / `Python <pythoncustomtr_>`_       | `Video Tutorial <videocustomtr_>`_     |
+----+------------------------------------------+--------------------------+--------------------------------------------------------------------+----------------------------------------+
| 2  |  *Dataset Generator*                     |       |dgenerator|       | `Notebook <ipythondgenerator_>`_ / `Python <pythondgenerator_>`_   | `Video Tutorial <videodgenerator_>`_   |
+----+------------------------------------------+--------------------------+--------------------------------------------------------------------+----------------------------------------+
| 3  |  *Create TFRecords*                      |       |tfrecords|        | `Notebook <ipythontfrecords_>`_ / `Python <pythontfrecords_>`_     | `Video Tutorial <videotfrecords_>`_    |
+----+------------------------------------------+--------------------------+--------------------------------------------------------------------+----------------------------------------+



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
  * Please do NOT change the ipython files. Instead, change the corresponsing PYTHON files.
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

========================
Developers
========================


**Company**: Instill AI [`Website
<https://instillai.com/>`_]

**Creator**: Machine Learning Mindset [`Blog
<https://machinelearningmindset.com/blog/>`_, `GitHub
<https://github.com/machinelearningmindset>`_, `Twitter
<https://twitter.com/machinemindset>`_]

**Developer**: Amirsina Torfi [`GitHub
<https://github.com/astorfi>`_, `Personal Website
<https://astorfi.github.io/>`_, `Linkedin
<https://www.linkedin.com/in/amirsinatorfi/>`_ ]
