Define New Waveforms
####################

Overview
========

The :mod:`~wavgen.waveform` module was designed to be extensible. By leveraging polymorphism, we can coerce a uniformity
in treatment across various waveform types. The resulting framework furnishes **Users** with (virtually) painless
procedure to define custom parameterized waveforms which cooperate with the core routines for AWG interaction &
:doc:`parallel processing <../info/parallel>`.

In short, a User implements a `class` to describe his waveform; appending it to the :mod:`~wavgen.waveform` source file.
To meet minimal integration requirements, a User `class` must:

Extend :class:`wavgen.waveform.Waveform`
    This `base class` represents a general waveform.
    Inheriting allows for consistency of treatment among all waveform types.

Appropriately initiate the base constructor
    The new constructor [#]_ must subsequently call the super constructor [#]_. It's mandatory you pass an `int` value
    for the ``sample_length`` argument, which configures the waveform's length in samples. The second argument, `amp`,
    can optionally receive a `float` value from 0 to 1, applied as an overall scaling factor to the final waveform.

Override the ``compute(self, p, q)`` method
    The heart of your definition; dictating how the waveform's samples are calculated.

Override the dual methods ``config_file`` & ``from_file``
    The former method facilitates a packaging of descriptive waveform parameters for file storage; which should be
    sufficient, when retrieved at a later time, for reconstruction of the waveform.
    Regarding the latter method, its `super` implementation is usually sufficient, obviating the need to override.

As a last resort, consult the :class:`~wavgen.waveform.Waveform` source documentation.

.. _this: https://rhettinger.wordpress.com/2011/05/26/super-considered-super/
.. [#] In python, ``__init__`` plays the role of constructor for a class.
.. [#] `Super` is another name for `parent` or `base` in terms of inheritance. Check this_ out

Example
=======

Below we present a modest example of a valid User defined class. We piece-wise analyze each of the methods overriding
an inherited method.

.. note::
   Variables in all CAPS are global values, being either a constant or parameter. See :mod:`~wavgen.config`

The example code aims at defining a humble **square wave**. Notice how ``SquareWave(Waveform)`` is `extending` the
``Waveform`` class::

    class SquareWave(Waveform):
        def __init__(self, f, arg1, arg2=0, optional_arg=None, sample_length=100, amp=1.0):
            self.Period = SAMP_FREQ / f
            self.Arg1 = arg1
            self.SampleLength = arg2 if optional_arg else sample_length

            super().__init__(self.SampleLength, amp)

        def compute(self, p, q):
            N = min(DATA_MAX, self.SampleLength - p*DATA_MAX)
            waveform = np.empty(N, dtype=float)

            for i in range(N):
                n = i + p*DATA_MAX
                phase = (n % self.Period) - self.Period/2
                waveform[i] = (1 if phase < 0 else -1)

            q.put((p, waveform))

        def config_file(self, h5py_f):
            ## Contents ##
            h5py_f.attrs.create('f', data=SAMP_FREQ / self.Period)
            h5py_f.attrs.create('arg1', data=self.Arg1)
            h5py_f.attrs.create('sample_length', data=self.SampleLength)

            ## Table of Contents ##
            h5py_f.attrs.create('keys', data=['arg1', 'f', 'sample_length'])

            return super().config_file(h5py_f)

        @classmethod
        def from_file(cls, **kwargs):
            return cls(**kwargs)

Overriding
==========

\_\_init\_\_(self, *anything*)
------------------------------
The User has nearly infinite freedom for creativity here.
Although you may want to consider how your choice impacts the :ref:`third <prev>` & :ref:`fourth <next>`
sub-sections below.

The only **real** requirement has already been mentioned above; namely, ``super().__init__(self.SampleLength, amp)``.
It doesn't quite matter how we determined ``self.SampleLength``, just that it exists and is an integer.

compute(self, p, q)
-------------------
This is the dispatch method used for :doc:`parallelization <../info/parallel>`.
In short:

- The waveform is divided into chunks of size ``DATA_MAX``, where the last chunk holds a remainder.
- ``p`` indicates which chunk to compute; which is stored in a `numpy array` of commensurate size.
- In final, we pair ``p`` & the `numpy array` in a tuple which is submitted to ``q``, an inter-process queue.
- All chunks are collected and ordered according to their ``p``, resulting in a monolithic array of the entire waveform.

If in doubt, follow this template which captures the aspects shared by most cases::

    N = min(DATA_MAX, self.SampleLength - p*DATA_MAX)  # Determines chunk size
    waveform = np.empty(N, dtype=float)                # Instantiates a numpy array

    for i in range(N):                                 # Iterate a relative index
        n = i + p*DATA_MAX                             # Derive an absolute index
        # something
        waveform[i] = # something                      # Calculate & store each absolute data point

    q.put((p, waveform))                               # Places results on the Queue

.. note::
    The `numpy array` is not restricted in terms of dtype, although it would seem that `float` type is probably
    always the optimal choice.

.. _prev:

config_file(self, h5py_f)
-------------------------
Raw waveform samples are saved in :ref:`HDF5 dataset <datasets>` structures; which is passed here as ``h5py_f``.
From this alone, it's not obvious how we'd determine the waveform class, let alone defining parameters. We address the
issue by attaching directly to the dataset a number of :ref:`attribute <attrs>` structures; composed of name & data
element, e.g. ``h5py_f.attrs.create("arg1", data=[1, 5, 7, 9])``.

There is freedom in implementation; the goal is to save enough information s.t. we can identify & reconstruct the
original waveform object, using only saved information.
A reliable technique is to choose a set of constructor arguments, through which you can effectively set each
class attribute. The example achieves such a subset, compare the method body::

    ## Contents ##
    h5py_f.attrs.create('f', data=SAMP_FREQ / self.Period)
    h5py_f.attrs.create('arg1', data=self.Arg1)
    h5py_f.attrs.create('sample_length', data=self.SampleLength)

To the class constructor::

    def __init__(self, f, arg1, arg2=0, optional_arg=None, sample_length=100, amp=1.0):
        self.Period = SAMP_FREQ / f
        self.Arg1 = arg1
        self.SampleLength = arg2 if optional_arg else sample_length

        super().__init__(self.SampleLength, amp)

Additionally, a mandatory `Table of Contents` attribute is created, holding an unordered list of all the attribute
keywords; it must be named ``'keys'`` as shown::

    ## Table of Contents ##
    h5py_f.attrs.create('keys', data=['arg1', 'f', 'sample_length'])

The list of keywords need not match the constructor's order.
(although it **does** need to considered in the :ref:`next <next>` sub-section).

Lastly you must end with ``return super().config_file(h5py_f)`` to process general formatting & return a handle on the
dataset.

.. _next:

from_file(cls, \*\*keys)
------------------------
This function is, in spirit, achieves the converse of :meth:`~wavgen.waveform.Waveform.config_file`.
It receives ``**keys``, a dictionary between keywords & HDF5 attribute values, ordered according to the keyword
``"keys"`` attribute, acting as our `Table of Contents`.

Most likely, you will be able to choose your ``**keys`` s.t. they each correspond to a constructor argument. In that
case, it is unnecessary to override this method's inherited form::

    @classmethod
    def from_file(cls, **kwargs):
        return cls(**kwargs)

For a terrific example of the contrary case, see the :class:`wavgen.waveform.Sweep` template.

.. attention::
    You need to put the `@classmethod` decorator above its function signature for somewhat unimportant reasons
    (see classmethod_ if curious!).

.. _classmethod: https://www.geeksforgeeks.org/class-method-vs-static-method-python/
