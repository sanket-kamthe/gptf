# gptf

**gptf** is a library for building Guassian Process models in Python using
[TensorFlow][tensorflow], based on [GPflow][GPflow]. Its benefits over
GPflow include:

- Ops can be easily pinned to devices / graphs, and inherit their device
  placement from their parents.
- Autoflow that plays nicely with the distributed runtime.
- Better trees for a better world.

[![Build Status](https://travis-ci.org/sanket-kamthe/gptf.svg?branch=master)](https://travis-ci.org/sanket-kamthe/gptf)
[![Coverage Status](https://coveralls.io/repos/github/sanket-kamthe/gptf/badge.svg?branch=master)](https://coveralls.io/github/sanket-kamthe/gptf?branch=master)

Explanatory notebooks can be found in the [notebooks directory][notebooks],
and documentation can be found [here][documentation].

## Installation

1. [Install TensorFlow manually][install tensorflow] (at least version 0.9).
2. Run `python setup.py develop`.

## Running tests

Tests can be run using `python setup.py nosetests`.


[tensorflow]: https://www.tensorflow.org
[install tensorflow]: https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#pip-installation
[GPflow]: https://github.com/GPflow/GPflow
[notebooks]: notebooks
[documentation]: http://icl-sml.github.io/gptf/
