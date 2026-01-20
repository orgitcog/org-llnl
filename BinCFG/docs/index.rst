Welcome to BinCFG's documentation!
==================================

BinCFG is A Python tool designed to parse binary analyzer outputs to produce Control Flow Graphs 
(CFGs), tokenize and normalize the assembly lines within those CFGs, and convert that data into ML-ready formats.

Installation Instructions
-------------------------

BinCFG can be installed with pip:
`pip install bincfg`

Overview
--------

BinCFG allows users to load in output from binary analysis tools, and convert them into python CFG objects. For
example, loading an output from the Rose binary analysis tool may look like:

.. highlight:: python
.. code-block:: python

   from bincfg import CFG

   cfg = CFG("/path/to/rose/cfg/file")

Now that you have a cfg object, you can print it to get some stats, or print its `.functions` attribute to print all
the functions in this CFG:

.. highlight:: python
.. code-block:: python

   print(cfg)
   print()
   print(cfg.functions)

   # Output:
   #
   # CFG with no normalizer and 14 functions, 56 basic blocks, 65 edges, and 219 lines of assembly
   #
   # [CFGFunction innerfunc '_init' at 0x00001000 with 3 blocks, 8 asm lines, called by 1 basic blocks across 1 functions,
   #  CFGFunction innerfunc with no name at 0x00001040 with 1 blocks, 2 asm lines, called by 1 basic blocks across 1 functions,
   #  CFGFunction externfunc 'printf@plt' at 0x00001050 with 1 blocks, 2 asm lines, called by 3 basic blocks across 1 functions,
   # ...

Often, you will wish to normalize the assembly lines in your CFG. This can be done by calling
the ``.normalize()`` function.

.. highlight:: python
.. code-block:: python

   norm_cfg = cfg.normalize('innereye')

   print(norm_cfg)
   print()
   print(cfg.functions[0].blocks)
   print()
   print(norm_cfg.functions[0].blocks)

   # NOTE: this is using the builtin normalization method based on the 'innereye' method
   #
   # Output:
   #
   # CFG with normalizer: 'InnerEyeNormalizer' and 14 functions, 56 basic blocks, 65 edges, and 219 lines of assembly
   #
   # [CFGBasicBlock in function "_init" at 0x00001000 unlabeled with 5 lines of assembly:
   #     0x00001000: nop
   #     0x00001004: sub    rsp, 0x08
   #     0x00001008: mov    rax, qword ds:[rip + 0x0000000000002fd9<12249,absolute=0x0000000000003fe8>]
   #     0x0000100f: test   rax, rax
   #     0x00001012: je     0x0000000000001016<4118>,
   #  CFGBasicBlock in function "_init" at 0x00001014 unlabeled with 1 lines of assembly:
   #     0x00001014: call   rax,
   #  CFGBasicBlock in function "_init" at 0x00001016 unlabeled with 2 lines of assembly:
   #     0x00001016: add    rsp, 0x08
   #     0x0000101a: ret]
   #
   # [CFGBasicBlock in function "_init" at 0x00001000 unlabeled with 5 lines of assembly:
   #     0x00001000: ['nop']
   #     0x00001004: ['sub_rsp_immval']
   #     0x00001008: ['mov_rax_[_rip_+_immval_]']
   #     0x0000100f: ['test_rax_rax']
   #     0x00001012: ['je_immval'],
   #  CFGBasicBlock in function "_init" at 0x00001014 unlabeled with 1 lines of assembly:
   #     0x00001014: ['call_func'],
   #  CFGBasicBlock in function "_init" at 0x00001016 unlabeled with 2 lines of assembly:
   #     0x00001016: ['add_rsp_immval']
   #     0x0000101a: ['ret']]

Basic blocks have now had their assembly lines normalized using the 'innereye' method.

This python representation of a CFG is relatively high-memory, inefficient, and not suitable for machine learning
purposes. Instead, it is intended to be used only for loading in and parsing output from binary analysis tools, as well
as providing an easier-to-traverse object if you wish to walk through or edit the CFG.

For a more memory/space efficient, ML-ready representation of a CFG, you can convert it to a ``MemCFG``:

.. highlight:: python
.. code-block:: python

   from bincfg import MemCFG

   mem_cfg = MemCFG(norm_cfg)

   print(mem_cfg)

   # Output:
   # 
   # MemCFG with normalizer: 'InnerEyeNormalizer' and 14 functions, 56 blocks, 219 assembly lines, and 65 edges

NOTE: ``MemCFG``'s require a normalization method. If you wish to use a ``MemCFG`` without doing a lossy normalization, you
can use the 'unnormalized' normalization provided by default by that architecture's ``BaseNormalizer``.

Finally, you can get ML-ready data using ``MemCFG`` functions/attributes. For example, the 
:func:`~bincfg.mem_cfg.MemCFG.to_adjacency_matrix` function will give you the adjacency matrix of that ``MemCFG`` in one
of multiple common forms. See the ``MemCFG`` object for more info.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   bincfg


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
