# convolutional_autoencoder

The original author is no longer making changes to this repo, but will review/accept pull requests...

Code for a convolutional autoencoder written on python, theano, lasagne, nolearn

I highly recommend you use the ipython notebook to run this, if you just need code to read, look at the python file.

Due to the fact that the newest versions of theano, nolearn and lasagne are sometimes in conflict, it is recommended that you work in a virtualenv (see http://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/).  Then you can install a set of modules that are compatible using pip install -r https://github.com/mikesj-public/convolutional_autoencoder/blob/master/requirements.txt

The newest version of lasagne makes use of cudNN.  If you have a decent graphics card you should follow the instructions at https://developer.nvidia.com/cudnn to take advantage of this (the code checks for it).
