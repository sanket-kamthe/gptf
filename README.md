# GPTF

GPTF is a library for building Guassian Process models in Python using
[TensorFlow][tensorflow], based on [GPflow][GPflow]. Its benefits over
GPflow include:

- Ops can be easily pinned to devices / graphs, and inherit their device
  placement from their parents.
- Autoflow that plays nicely with the distributed runtime.
- Better trees for a better world.

# Installation

If you are running Python 2.7, 3.4 or 3.5 on Linux or OS X, and you only
need the CPU version of TensorFlow, installing is as simple as 
`python setup.py develop`. For other use cases, you will need to 
[install TensorFlow manually][install tensorflow] (at least version 0.9).
Once TensorFlow is installed, run `python setup.py develop`.

# Running tests

Tests can be run using `python setup.py nosetests`.


[tensorflow]: https://www.tensorflow.org
[install tensorflow]: https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#pip-installation
[GPflow]: https://github.com/GPflow/GPflow
