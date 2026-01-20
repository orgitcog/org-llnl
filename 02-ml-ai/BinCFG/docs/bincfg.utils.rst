bincfg.utils package
====================

Various utility functions and objects.

AtomicTokenDict
---------------

When doing multithreaded processing with BinCFG, it would be useful to have the ability to do atomic synchronized updates
of the current tokens that are being used when normalizing (that way, all ``MemCFG``s use the same shared tokens).
The ``AtomicTokenDict`` allows for atomic updates to a shared token dictionary requiring only a shared filesystem
to work. It ensures only one process can update a pickle file containing the token dictionary at a time using the
`atomicwrites` pip package.

There are a couple possible downsides depending on how you use it:

1. If you are doing a bunch of updates at the same time, that can be really slow. It may help to precompute much of
   the common tokens initially before doing a large multithreaded/HPC run to help get over this initial hurdle
2. Crashing/interrupted code can cause deadlocks if they stop execution while the AtomicTokenDict is updating. If this
   occurs, you can delete the lockfile ('.[filename].lock' where '[filename]' is the name of the pickle file), and that
   fixes it

Submodules
----------

bincfg.utils.atomic\_token\_dict module
---------------------------------------

.. automodule:: bincfg.utils.atomic_token_dict
   :members:
   :undoc-members:
   :show-inheritance:

bincfg.utils.cfg\_utils module
------------------------------

.. automodule:: bincfg.utils.cfg_utils
   :members:
   :undoc-members:
   :show-inheritance:

bincfg.utils.misc\_utils module
-------------------------------

.. automodule:: bincfg.utils.misc_utils
   :members:
   :undoc-members:
   :show-inheritance:

bincfg.utils.type\_utils module
-------------------------------

.. automodule:: bincfg.utils.type_utils
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: bincfg.utils
   :members:
   :undoc-members:
   :show-inheritance:
