####################
Define New Waveforms
####################

.. _define:

Overview
--------

The :mod:`~wavgen.waveform` module was designed to make defining new user waveforms as simple as possible,
while still utilizing all the benefits of inherited functions & :ref:`parallel processing <parallel>`.

Since definitions are made by extending a generalized base class (:class:`wavgen.waveform.Waveform`),
then a user must write a class, appending to the :mod:`~wavgen.waveform` python file,
which overrides a number of minimally required methods.
These instructions will describe the minimal requirements, as well as where you can expand,
regarding the new waveform's definition.
Additionally, by reading the :class:`~wavgen.waveform.Waveform` documentation one can find
descriptions under each method regarding how to override.

Instructions
------------
First briefly examine the example below, then after are itemized instructions regarding each
individual overridden function.

Example
"""""""
This demonstrates extending the :class:`~wavgen.waveform.Waveform` base class to define
a **square wave**. We full-fill just a bit more than the bare minimum requirements (to depict exceptions).::

    class SquareWave(Waveform):
        def __init__(self, f, arg1, arg2=0, unused=0, sample_length=100):
            self.Period = SAMP_FREQ / f
            self.Arg1 = arg1

            super().__init__(sample_length)

        def compute(self, p, q):
            N = min(DATA_MAX, self.SampleLength - p*DATA_MAX)
            waveform = np.empty(N, dtype='int16')

            for i in range(N):
                n = i + p*DATA_MAX
                phase = (n % self.Period) - self.Period/2
                waveform[i] = int(SAMP_VAL_MAX * (1 if phase < 0 else -1))

            q.put((p, waveform))

        def config_file(self, h5py_f):
            ## Table of Contents ##
            h5py_f.attrs.create('attrs', data=['arg1', 'f', 'arg2'])

            ## Contents ##
            h5py_f.attrs.create('f', data=SAMP_FREQ / self.Period)
            h5py_f.attrs.create('arg1', data=self.Arg1)
            h5py_f.attrs.create('arg2', data=np.arange(5))

            return super().config_file(h5py_f)

        @classmethod
        def from_file(cls, *attrs):
            arg1, f, arg2, sample_length = attrs
            return cls(f, arg1, arg2=arg2, sample_length=sample_length)

\_\_init\_\_(self, *anything*)
""""""""""""""""""""""""""""""
Here you define all of the waveform's describing parameters as instance variables.
No constructor arguments are necessary. What *is* required is that an `int` (`sample_length` in the example)
must be passed to the parent constructor, no matter how you get you `int`::

    super().__init__(sample_length)

compute(self, p, q)
"""""""""""""""""""
This is the dispatch method for child processes (:ref:`parallelism <parallel>`),
meaning each spawned child runs only this method, though each has a different value for `p`.
To compute an entire waveform, it is first split into chunks of size :const:`wavgen.waveform.DATA_MAX`,
where the last chunk holds the remainder.
The chunks are then indexed, **starting at 0** by `p`, so the calculation must be done accordingly.
This method will probably **always** take the form::

    N = min(DATA_MAX, self.SampleLength - p*DATA_MAX)
    waveform = np.empty(N, dtype='int16')

    for i in range(N):
        n = i + p*DATA_MAX
        # something
        waveform[i] = # something

    q.put((p, waveform))

This depicts the absolute essentials.

.. _prev:

config_file(self, h5py_f)
"""""""""""""""""""""""""
Here is the method which prepares the meta-data & structure of the :ref:`HDF5 <hdf5>`
file where the waveform is saved when requested.
The first important piece::

    ## Table of Contents ##
    h5py_f.attrs.create('attrs', data=['arg1', 'f', 'arg2'])

    ## Contents ##
    h5py_f.attrs.create('f', data=SAMP_FREQ / self.Period)
    h5py_f.attrs.create('arg1', data=self.Arg1)
    h5py_f.attrs.create('arg2', data=np.arange(5))

must be present, where the `data=` argument must be a `list of str` where
each string is the name of a necessary input argument, in order, to construct the correct
object with the class constructor; they are saved as attributes on the file.
Then you must save each value as shown under `## Contents ##`[#]_.
E.g. notice how the above corresponds to the constructor signature::

    def __init__(self, f, arg1, arg2=0, unused=0, sample_length=100):

The arguments may have defaults and the order of the `data=` list need not match
the order of arguments on `__init__` (although it **must** be considered in the :ref:`next <next>` sub-section).
The constructor could even sport additional arguments not included in the list
**as long as they are optional** (E.g. the `unused=` argument).

There is some limit to how you can go about this,
which will become clear in the :ref:`following <next>` sub-section.

Second important piece::

    return super().config_file(h5py_f)

This calls the generalized parent method, returning an object which acts as a handle
to the :ref:`dataset <dataset>` where the waveform samples are written.

.. _next:

from_file(cls, \*attrs)
"""""""""""""""""""""""
This is the waveform specific file called by the module level :func:`~wavgen.waveform.from_file`
for extracting waveform specific attributes from :ref:`HDF5 <hdf5>` files
& marshalling them into the proper constructor.
You need to put the `@classmethod` decorator above its function signature for somewhat unimportant reasons
(see classmethod_ if curious!).
This function is, in spirit, the inverse of :meth:`~wavgen.waveform.Waveform.config_file`,
and thus sets the limit on how complicated :meth:`~wavgen.waveform.Waveform.config_file` can be
(mentioned :ref:`above <prev>`).
The input argument `*attrs` will be a tuple containing each of the
Following our running example from the :ref:`previous sub-section <prev>`

By comparing to the constructor, you can see how this would create the proper object.

.. _classmethod: https://www.geeksforgeeks.org/class-method-vs-static-method-python/

.. [#] For rules on what types the values can take, see :ref:`this <attrs>` guide.