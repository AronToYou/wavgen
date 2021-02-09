Modify These Docs
#################

The documentation is produced using a tool called Sphinx_, which just extends the reStructuredText_ (reST) markup language with lots of useful extras.
If you want to modify things, I'd following the link which bares its name and use it to decrypt the most simple source files in the :file:`/wavgen/docs/source/` directory.

A good start is the :file:`source/index.rst` file, which defines the root page of the documentation.

Once you have made your changes and/or added new source files, you can build the documentation with this :file:`../build.ps1` Powershell script. The script does just this:
1. Tells Sphinx to build, which annoyingly builds into a new directory ``/wavgen/docs/source/html/``
2. Deletes the old build. (Everthing in ``/wavgen/docs/`` except for ``/docs/source/``)
3. Moves everything inside of ``/html/`` into ``/wavgen/docs/``
4. Deletes the now empty ``/html/`` directory

Blam!

.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html
.. _Sphinx: https://www.sphinx-doc.org/en/master/