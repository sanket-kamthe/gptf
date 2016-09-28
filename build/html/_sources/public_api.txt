Public API
==========

.. toctree::
   :maxdepth: 3

Parameterized classes
---------------------

To create a new parameterized class, subclass :py:class:`gptf.Parameterized`
and one of :py:class:`gptf.ParamAttributes` or :py:class:`gptf.ParamList`,
depending on how you want parameters to be accessed.

Methods of parameterized objects that create tensorflow objects should
be decorated with :py:function:`gptf.tf_method()`.

.. autoclass:: gptf.Parameterized
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: gptf.ParamAttributes
   :members:
   :undoc-members:

.. autoclass:: gptf.ParamList
   :members:
   :undoc-members:

.. autoclass:: gptf.Param
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: gptf.DataHolder
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: gptf.tf_method

Autoflow
--------

The autoflow decorator allows you to define a method of a parameterized class
using Tensorflow, but interact with it using NumPy. It applies 
`gptf.tf_method()` for you.

.. autofunction:: gptf.autoflow

Models
------

These abstract base classes implement models that can be optimised. Subclass
these and implement the abstract methods to implement new models.
See GPR_ for more information.

.. autoclass:: gptf.Model
   :members:
   :undoc-members:

.. autoclass:: gptf.GPModel
   :members:
   :undoc-members:

Public core modules
-------------------

The following core modules are part of the public API, and are available
under `gptf.<module name>` as well as `gptf.core.<module name>`, e.g.

>>> import gptf
>>> from gptf.core import kernels
>>> gptf.kernels is kernels
True

densities
^^^^^^^^^

.. automodule:: gptf.core.densities
   :noindex:
   :members:
   :undoc-members:

transforms
^^^^^^^^^^

.. automodule:: gptf.core.transforms
   :noindex:
   :members:
   :undoc-members:

kernels
^^^^^^^

All classes in this module inherit from :py:class:`gptf.Parameterized`.

.. automodule:: gptf.core.densities
   :noindex:
   :members:
   :undoc-members:

likelihoods
^^^^^^^^^^^

All classes in this module inherit from :py:class:`gptf.Parameterized`.

.. automodule:: gptf.core.likelihoods
   :noindex:
   :members:
   :undoc-members:

meanfunctions
^^^^^^^^^^^^^

All classes in this module inherit from :py:class:`gptf.Parameterized`.

.. automodule:: gptf.core.meanfunctions
   :noindex:
   :members:
   :undoc-members:

GPR
---

For example code that plays with the classes in this module, see the 
`GPR notebook`_.

.. automodule:: gptf.gpr
   :noindex:
   :members:
   :undoc-members:

Distributed
-----------

For example code that plays with the classes in this module, see the
`distributed GP models`_ and the `distributed computation`_ notebooks.

.. automodule:: gptf.distributed
   :noindex:
   :members:
   :undoc-members:

.. _GPR notebook: https://github.com/ICL-SML/gptf/blob/master/notebooks/Gaussian_process_regression.ipynb
.. _distributed GP models: https://github.com/ICL-SML/gptf/blob/master/notebooks/Distributed Gaussian process models.ipynb
.. _distributed computation: https://github.com/ICL-SML/gptf/blob/master/notebooks/Distributed computation.ipynb
