wavgen! |bounce|
################

A Python package to simplify communication with a :doc:`Spectrum AWG card <info/spectrum>`, for the use & purposes
of the E6_ experiment.

.. |bounce| image:: _static/ball1.gif

What it Offers
==============

+ Calculate, plot, & store custom parameterized waveforms
+ Optimize *power* & *homogeneity* of **static** microtrap configurations
+ Define sequences of waveforms capable of:

    + Conditioning step transition on a software or hardware trigger
    + Re-definition during operation

.. Warning::
    The last two above listed features are still pre-alpha.

.. DANGER::
    There have been a couple large design changes in the code which don't agree with certain points in the how-to pages. The docs section is procedurally generated from the code itself, including the extensive inline ``docstring`` descriptions.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   Github <https://github.com/AronToYou/wavgen>
   how-to/how-to.rst
   info/info.rst
   docs/docs.rst
   other


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _E6: http://ultracold.physics.berkeley.edu/e6-cavity-microscope
