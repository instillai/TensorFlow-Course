==================================
Install TensorFlow from the source
==================================

.. _TensorFlow: https://www.tensorflow.org/install/
.. _Installing TensorFlow from Sources: https://www.tensorflow.org/install/install_sources
.. _Bazel Installation: https://bazel.build/versions/master/docs/install-ubuntu.html
.. _CUDA Installation: https://github.com/astorfi/CUDA-Installation
.. _NIDIA documentation: https://github.com/astorfi/CUDA-Installation



The installation is available at `TensorFlow`_. Installation from the source is recommended because the user can build the desired TensorFlow binary for the specific architecture. It enriches the TensoFlow with a better system compatibility and it will run much faster. Installing from the source is available at `Installing TensorFlow from Sources`_ link. The official TensorFlow explanations are concise and to the point. However. few things might become important as we go through the installation. We try to project the step by step process to avoid any confusion. The following sections must be considered in the written order.

The assumption is that installing TensorFlow in the ``Ubuntu`` using ``GPU support`` is desired. ``Python2.7`` is chosen for installation.

**NOTE** Please refer to this youtube `link <youtube_>`_ for a visual explanation.

.. _youtube: https://www.youtube.com/watch?v=_3JFEPk4qQY&t=2s

------------------------
Prepare the environment
------------------------

The following should be done in order:
 
    * TensorFlow Python dependencies installation
    * Bazel installation
    * TensorFlow GPU prerequisites setup

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TensorFlow Python Dependencies Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For installation of the required dependencies, the following command must be executed in the terminal:

.. code:: bash

    sudo apt-get install python-numpy python-dev python-pip python-wheel python-virtualenv
    sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel python3-virtualenv
    
The second line is for ``python3`` installation.

~~~~~~~~~~~~~~~~~~~
Bazel Installation
~~~~~~~~~~~~~~~~~~~

Please refer to `Bazel Installation`_.

``WARNING:`` The Bazel installation may change the supported kernel by the GPU! After that you may need to refresh your GPU installation or update it, otherwise, you may get the following error when evaluating the TensorFlow installation:

.. code:: bash

    kernel version X does not match DSO version Y -- cannot find working devices in this configuration
    
For solving that error you may need to purge all NVIDIA drivers and install or update them again. Please refer to `CUDA Installation`_ for further detail.


    
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TensorFlow GPU Prerequisites Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following requirements must be satisfied:

    * NVIDIA's Cuda Toolkit and its associated drivers(version 8.0 is recommended). The installation is explained at `CUDA Installation`_.
    * The cuDNN library(version 5.1 is recommended). Please refer to `NIDIA documentation`_ for further details.
    * Installing the ``libcupti-dev`` using the following command: ``sudo apt-get install libcupti-dev``

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Creating a Virtual Environment (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assume the installation of TensorFlow in a ``python virtual environment`` is desired. First, we need to create a directory to contain all the environments. It can be done by executing the following in the terminal:

.. code:: bash

    sudo mkdir ~/virtualenvs

Now by using the ``virtualenv`` command, the virtual environment can be created:

.. code:: bash

    sudo virtualenv --system-site-packages ~/virtualenvs/tensorflow

**Environment Activation**

Up to now, the virtual environment named *tensorflow* has been created. For environment activation, the following must be done:

.. code:: bash

    source ~/virtualenvs/tensorflow/bin/activate

However, the command is too verbose! 

**Alias**

The solution is to use an alias to make life easy! Let's execute the following command:

.. code:: bash

    echo 'alias tensorflow="source $HOME/virtualenvs/tensorflow/bin/activate" ' >> ~/.bash_aliases
    bash

After running the previous command, please close and open terminal again. Now by running the following simple script, the tensorflow environment will be activated.

.. code:: bash

    tensorflow
    
**check the ``~/.bash_aliases``**

To double check let's check the ``~/.bash_aliases`` from the terminal using the ``sudo gedit ~/.bash_aliases`` command. The file should contain the following script:

.. code:: shell

    alias tensorflow="source $HO~/virtualenvs/tensorflow/bin/activate" 
    

**check the ``.bashrc``**

Also, let's check the ``.bashrc`` shell script using the ``sudo gedit ~/.bashrc`` command. It should contain the following:
 
.. code:: shell

    if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
    fi
 

    
---------------------------------
Configuration of the Installation
---------------------------------

At first, the Tensorflow repository must be cloned:

.. code:: bash

     git clone https://github.com/tensorflow/tensorflow 

After preparing the environment, the installation must be configured. The ``flags`` of the configuration are of great importance because they determine how well and compatible the TensorFlow will be installed!! At first, we have to go to the TensorFlow root:

.. code:: bash

    cd tensorflow  # cd to the cloned directory

The flags alongside with the configuration environment are demonstrated below:

.. code:: bash

    $ ./configure
    Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python2.7
    Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:
    Do you wish to use jemalloc as the malloc implementation? [Y/n] Y
    jemalloc enabled
    Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] N
    No Google Cloud Platform support will be enabled for TensorFlow
    Do you wish to build TensorFlow with Hadoop File System support? [y/N] N
    No Hadoop File System support will be enabled for TensorFlow
    Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] N
    No XLA JIT support will be enabled for TensorFlow
    Found possible Python library paths:
      /usr/local/lib/python2.7/dist-packages
      /usr/lib/python2.7/dist-packages
    Please input the desired Python library path to use.  Default is [/usr/local/lib/python2.7/dist-packages]
    Using python library path: /usr/local/lib/python2.7/dist-packages
    Do you wish to build TensorFlow with OpenCL support? [y/N] N
    No OpenCL support will be enabled for TensorFlow
    Do you wish to build TensorFlow with CUDA support? [y/N] Y
    CUDA support will be enabled for TensorFlow
    Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:
    Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave empty to use system default]: 8.0
    Please specify the location where CUDA 8.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
    Please specify the cuDNN version you want to use. [Leave empty to use system default]: 5.1.10
    Please specify the location where cuDNN 5 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
    Please specify a list of comma-separated Cuda compute capabilities you want to build with.
    You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
    Please note that each additional compute capability significantly increases your build time and binary size.
    [Default is: "3.5,5.2"]: "5.2"


**NOTE:**
     * The cuDNN version must be exactly determined using the associated files in /usr/local/cuda
     * The compute capability is spesified related the ``available GPU model`` in the system architecture. For example ``Geforce GTX Titan X`` GPUs have compute capability of 5.2.
     * Using ``bazel clean`` is recommended if re-configuration is needed.

**WARNING:**
     * In case installation of the TwnsorFlow in a virtual environment is desired, the environment must be activated at first and before running the ``./configure`` script.
     
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Test Bazel (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can run tests using ``Bazel`` to make sure everything's fine:

.. code:: bash

    ./configure
    bazel test ...

---------------------
Build the .whl Package
---------------------

After configuration of the setup, the pip package needs to be built by the Bazel.
    
To build a TensorFlow package with GPU support, execute the following command:

.. code:: bash

    bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
    
The ``bazel build`` command builds a script named build_pip_package. Running the following script build a .whl file within the ~/tensorflow_package directory:

.. code:: bash

    bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/tensorflow_package





-------------------------------
Installation of the Pip Package
-------------------------------

Two types of installation can be used. The native installation using system root and the virtual environment installation.

~~~~~~~~~~~~~~~~~~~~~~~~~~~
Native Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following command will install the pip package created by Bazel build:

.. code:: bash

    sudo pip install ~/tensorflow_package/file_name.whl
    

~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using Virtual Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

At first, the environment must be activation. Since we already defined the environment alias as ``tensorflow``, by the terminal execution of the simple command of ``tensorflow``, the environment will be activated. Then like the previous part, we execute the following:

.. code:: bash
    
    pip install ~/tensorflow_package/file_name.whl

**WARNING**:
           * By using the virtual environment installation method, the sudo command should not be used anymore because if we use sudo, it points to native system packages and not the one available in the virtual environment.
           * Since ``sudo mkdir ~/virtualenvs`` is used for creating of the virtual environment, using the ``pip install`` returns ``permission error``. In this case, the root privilege of the environment directory must be changed using the ``sudo chmod -R 777 ~/virtualenvs`` command.
    
--------------------------
Validate the Installation
--------------------------

In the terminal, the following script must be run (``in the home directory``) correctly without any error and preferably any warning:

.. code:: bash

    python
    >> import tensorflow as tf
    >> hello = tf.constant('Hello, TensorFlow!')
    >> sess = tf.Session()
    >> print(sess.run(hello))

--------------------------
Common Errors
--------------------------

Different errors reported blocking the compiling and running TensorFlow.

   * ``Mismatch between the supported kernel versions:`` This error mentioned earlier in this documentation. The naive solution reported being the reinstallation of the CUDA driver.
   * ``ImportError: cannot import name pywrap_tensorflow:`` This error usually occurs when the Python loads the tensorflow libraries from the wrong directory, i.e., not the version installed by the user in the root. The first step is to make sure we are in the system root such that the python libraries are utilized correctly. So basically we can open a new terminal and test TensorFlow installation again. 
   * ``ImportError: No module named packaging.version":`` Most likely it might be related to the ``pip`` installation. Reinstalling that using ``python -m pip install -U pip`` or ``sudo python -m pip install -U pip`` may fix it!

--------------------------
Summary
--------------------------

In this tutorial, we described how to install TensorFlow from the source which has the advantage of more compatibility with the system configuration. Python virtual environment installation has been investigated as well to separate the TensorFlow environment from other environments. Conda environments can be used as well as Python virtual environments which will be explained in a separated post. In any case, the TensorFlow installed from the source can be run much faster than the pre-build binary packages provided by the TensorFlow although it adds the complexity to installation process.




