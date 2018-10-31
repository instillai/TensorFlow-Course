=========================
Multi Layer Perceptron
=========================

This code is developed for training a ``Multi Layer Perceptron`` architecture in which the input will be feed-forwarded to the network that contains some hidden layers.

.. figure:: https://github.com/astorfi/TensorFlow-World/blob/master/docs/_img/3-neural_network/multi-layer-perceptron/neural-network.png
   :scale: 50 %
   :align: center
   

--------
Training
--------

**Train:**

The training can be run using the **train.sh** `bash script` file using the following command:

.. code-block:: bash

   ./train.sh

The bash script is as below:


.. code-block:: bash

   python train_mlp.py \
     --batch_size=512 \
     --max_num_checkpoint=10 \
     --num_classes=10 \
     --num_epochs=1 \
     --initial_learning_rate=0.001 \
     --num_epochs_per_decay=1 \
     --is_training=True \
     --allow_soft_placement=True \
     --fine_tuning=False \
     --online_test=True \
     --log_device_placement=False

**helper:**

In order to realize that what are the parameters as input running the following command is recommended:

.. code-block:: bash

   python train_mlp.py --help


In which `train_mlp.py` is the main file for running the training. The result of the above command will be as below:

.. code-block:: bash

  --train_dir TRAIN_DIR
                        Directory where event logs are written to.
  --checkpoint_dir CHECKPOINT_DIR
                        Directory where checkpoints are written to.
  --max_num_checkpoint MAX_NUM_CHECKPOINT
                        Maximum number of checkpoints that TensorFlow will
                        keep.
  --num_classes NUM_CLASSES
                        Number of model clones to deploy.
  --batch_size BATCH_SIZE
                        Number of model clones to deploy.
  --num_epochs NUM_EPOCHS
                        Number of epochs for training.
  --initial_learning_rate INITIAL_LEARNING_RATE
                        Initial learning rate.
  --learning_rate_decay_factor LEARNING_RATE_DECAY_FACTOR
                        Learning rate decay factor.
  --num_epochs_per_decay NUM_EPOCHS_PER_DECAY
                        Number of epoch pass to decay learning rate.
  --is_training [IS_TRAINING]
                        Training/Testing.
  --fine_tuning [FINE_TUNING]
                        Fine tuning is desired or not?.
  --online_test [ONLINE_TEST]
                        Fine tuning is desired or not?.
  --allow_soft_placement [ALLOW_SOFT_PLACEMENT]
                        Automatically put the variables on CPU if there is no
                        GPU support.
  --log_device_placement [LOG_DEVICE_PLACEMENT]
                        Demonstrate which variables are on what device.


-----------
Evaluation
-----------

The evaluation will be run using the **evaluation.sh** `bash script` file using the following command:

.. code-block:: bash

   ./evaluation.sh


