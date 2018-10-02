
Sections
~~~~~~~~

-  `Introduction <#Introduction>`__
-  `Description of the Overall
   Process <#Description%20of%20the%20Overall%20Process>`__
-  `How to Do It in Code? <#How%20to%20Do%20It%20in%20Code?>`__
-  `Summary <#Summary>`__

Linear Regression using TensorFlow
----------------------------------

This tutorial is about training a linear model by TensorFlow to fit the
data. Alternatively, you can check this `blog post <blogpostlinearregression_>`_.

.. _blogpostlinearregression: http://www.machinelearninguru.com/deep_learning/tensorflow/machine_learning_basics/linear_regresstion/linear_regression.html



Introduction
------------

In machine learning and statistics, Linear Regression is the modeling of
the relationship between a variable such as Y and at least one
independent variable as X. In the linear regression, the linear
relationship will be modeled by a predictor function which its
parameters will be estimated by the data and is called a Linear Model.
The main advantage of Linear Regression algorithm is its simplicity using
which it is very straightforward to interpret the new model and map the
data into a new space. In this article, we will introduce how to train a
linear model using TensorFLow and how to showcase the generated model.

Description of the Overall Process
----------------------------------

In order to train the model, the TensorFlow loops through the data and
it should find the optimal line (as we have a linear model) that fits
the data. The linear relationship between two variables of X, Y is
estimated by designing an appropriate optimization problem for which the requirement
 is a proper loss function. The dataset is available from the
`Stanford course CS
20SI <http://web.stanford.edu/class/cs20si/index.html>`__: TensorFlow
for Deep Learning Research.

How to Do It in Code?
---------------------

The process is started by loading the necessary libraries and the
dataset:

.. code:: python


    # Data file provided by the Stanford course CS 20SI: TensorFlow for Deep Learning Research.
    # https://github.com/chiphuyen/tf-stanford-tutorials
    DATA_FILE = "data/fire_theft.xls"

    # read the data from the .xls file.
    book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
    sheet = book.sheet_by_index(0)
    data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
    num_samples = sheet.nrows - 1

    #######################
    ## Defining flags #####
    #######################
    tf.app.flags.DEFINE_integer(
        'num_epochs', 50, 'The number of epochs for training the model. Default=50')
    # Store all elements in FLAG structure!
    FLAGS = tf.app.flags.FLAGS

Then we continue by defining and initializing the necessary variables:

.. code:: python

    # creating the weight and bias.
    # The defined variables will be initialized to zero.
    W = tf.Variable(0.0, name="weights")
    b = tf.Variable(0.0, name="bias")

After that, we should define the necessary functions. Different tabs
demonstrate the defined functions:

.. code:: python

    def inputs():
        """
        Defining the place_holders.
        :return:
                Returning the data and label lace holders.
        """
        X = tf.placeholder(tf.float32, name="X")
        Y = tf.placeholder(tf.float32, name="Y")
        return X,Y

.. code:: python

    def inference():
        """
        Forward passing the X.
        :param X: Input.
        :return: X*W + b.
        """
        return X * W + b

.. code:: python

    def loss(X, Y):
        """
        compute the loss by comparing the predicted value to the actual label.
        :param X: The input.
        :param Y: The label.
        :return: The loss over the samples.
        """

        # Making the prediction.
        Y_predicted = inference(X)
        return tf.squared_difference(Y, Y_predicted)

.. code:: python

    # The training function.
    def train(loss):
        learning_rate = 0.0001
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

Next, we are going to loop through different epochs of data and perform
the optimization process:

.. code:: python

    with tf.Session() as sess:

        # Initialize the variables[w and b].
        sess.run(tf.global_variables_initializer())

        # Get the input tensors
        X, Y = inputs()

        # Return the train loss and create the train_op.
        train_loss = loss(X, Y)
        train_op = train(train_loss)

        # Step 8: train the model
        for epoch_num in range(FLAGS.num_epochs): # run 100 epochs
            for x, y in data:
              train_op = train(train_loss)

              # Session runs train_op to minimize loss
              loss_value,_ = sess.run([train_loss,train_op], feed_dict={X: x, Y: y})

            # Displaying the loss per epoch.
            print('epoch %d, loss=%f' %(epoch_num+1, loss_value))

            # save the values of weight and bias
            wcoeff, bias = sess.run([W, b])

In the above code, the sess.run(tf.global\_variables\_initializer())
initialize all the defined variables globally. The train\_op is built
upon the train\_loss and will be updated in each step. In the end, the
parameters of the linear model, e.g., wcoeff and bias, will be returned.
For evaluation, the prediction line and the original data will be
demonstrated to show how the model fits the data:

.. code:: python

    ###############################
    #### Evaluate and plot ########
    ###############################
    Input_values = data[:,0]
    Labels = data[:,1]
    Prediction_values = data[:,0] * wcoeff + bias
    plt.plot(Input_values, Labels, 'ro', label='main')
    plt.plot(Input_values, Prediction_values, label='Predicted')

    # Saving the result.
    plt.legend()
    plt.savefig('plot.png')
    plt.close()

The result is depicted in the following figure:

.. figure:: https://github.com/astorfi/TensorFlow-World/blob/master/docs/_img/2-basics_in_machine_learning/linear_regression/updating_model.gif
   :scale: 50 %
   :align: center

**Figure 1:** The original data alongside with the estimated linear
model.

The above animated GIF shows the model with some tiny movements which
demonstrate the updating process. As it can be observed, the linear
model is not certainly among the bests! However, as we mentioned, its
simplicity is its advantage!

Summary
-------

In this tutorial, we walked through the linear model creation using
TensorFlow. The line which was found after training is not guaranteed
to be the best one. Different parameters affect the convergence
accuracy. The linear model is found using stochastic optimization and
its simplicity makes our world easier.
