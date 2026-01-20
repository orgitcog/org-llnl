bincfg.normalization package
============================

This subpackage provides classes to tokenize and normalize assembly lines, as well as the ability to easily create new
tokenization/normalization methods.

This library currently supports the following architectures:

   - x86/x86\_64
   - java

And disassembler output from the following binary analysis tools:

   - Rose https://github.com/rose-compiler/rose

``Tokenizer`` classes convert assembly instructions into lists of individual tokens for later processing. ``Normalizer``
classes take those tokens and normalize them to create the final string tokens for later use in models. This normalization
process is useful to prevent overfitting and Out of Vocabulary (OOV) problems in machine learning models.

An example of using a default ``X86BaseNormalizer`` on some x86\_64 assembly:

.. highlight:: python
.. code-block:: python

   from bincfg.normalization import X86BaseNormalizer

   asm_lines = [
      '0x00402cdd: add    rsp, 0x08',
      '0x00402cf0: push   qword ds:[rip + 0x0000000000252312<absolute=0x0000000000655008>]',
      'CALL   0x0000000000403360'
   ]
   normalizer = X86BaseNormalizer()

   for line in asm_lines:
      print(normalizer.normalize(line))

Which would give the output:

   >>> add rsp 8
   >>> push qword [ rip + 2433810 ]
   >>> call 4207456

The ``BaseNormalizer`` classes by default do some simple cleaning while keeping all of the necessary information for the
assembly line itself. For example: removing memory addresses of the instruction itself if it exists, converting all values
to decimal, removing extra whitespace/commas, etc.

This process is split into two main parts: `tokenization`, and `normalization`.

Tokenization
------------




Normalization
-------------

``Normalizer`` classes will normalize incoming strings. They do this by first tokenizing the strings (using either a
user-defined or default tokenizer), then normalizing that stream of (token\_name, token\_string) tuples into strings.

Normalization has two possible `Tokenization Levels` for the incoming strings:

   - 'op': opcode/operand level tokenization. Each individual opcode/operand gets normalized into its own token
   - 'instruction': instruction level tokenization. Each instruction line gets normalized into a single token, with
     all opcodes/operands in that instruction joined together, separated by some separator string (defaults to ' ' for
     ``BaseNormalizer``, and '_' for all other normalizers)

This library has a few built-in normalization methods based on literature:

   - InnerEye: https://arxiv.org/pdf/1808.04706.pdf
   - Deep Bin Diff: https://www.ndss-symposium.org/wp-content/uploads/2020/02/24311-paper.pdf
   - SAFE: https://github.com/gadiluna/SAFE
   - Deep Semantic: https://arxiv.org/abs/2106.05478

This module also provides a :func:`~bincfg.normalization.normalize.normalize_cfg_data` function to normalize CFG data.

Custom Normalizers
------------------

Creating custom normalizers is quite simple. In fact, multiple of the built-in normalization techniques are as simple
as a few lines of code:

.. highlight:: python
.. code-block:: python

   class X86InnerEyeNormalizer(X86BaseNormalizer):
      DEFAULT_TOKENIZATION_LEVEL = TokenizationLevel.INSTRUCTION
      handle_immediate = return_immstr(include_negative=True)
      handle_memory_size = ignore
      handle_function_call = replace_function_call_immediate(FUNCTION_CALL_STR)

Custom normalizers should inherit from ``BaseNormalizer``, and override parent methods to alter functionality. Most
methods do exactly as they say, "handling" the tokens in their names:

   - `handle_opcode()`
   - `handle_memory_size()`
   - `handle_register()`
   - `handle_immediate()`
   - `handle_memory_expression()`
   - `handle_rose_info()`
   - `handle_ignored()`
   - `handle_mismatch()`

There are some handlers that have slightly different functionality:

   - `handle_newline()`: this gets called after each full string has been parsed, or a new line character was found,
     indicating the end of a single assembly instruction. The full instruction will then be parsed, modified if necessary,
     specific opcodes handled, and converted into the final string (or list of strings if using 'op' tokenization level).
   - `handle_instruction()`: this gets called by `handle_newline()`. It will parse the full instruction, checking for
     any specifc opcodes that need to be handled. This method does not do any other cleaning/converting of the instruction.

Specific opcodes can be handled differently after the full line has been parsed. The `register_opcode_handler()`
function allows you to pass in a string regular expression to identify the opcodes to handle, and a function to handle
those opcodes. There are also a few built-in opcode handler functions:

   - `handle_jump()`: handles jump instructions
   - `handle_call()`: handles call instructions
   - 'nop' instructions: all 'nop' instructions will have everything stripped from them except the 'nop' opcode itself,
     since there is often a large amount of useless/extraneous information alongside those filler instructions

Finally, one can add in behavior for brand new token types using the `handle_unknown_token()` method, which will have
passed to it the token\_name and token\_string whenever an unknown token\_name is found. This way, you need not create
an entirely new ``Normalizer`` class, and can still use ``BaseNormalizer`` as a parent, if you wish to add in new token
types to parse.

For info on method signatures/expected return values, see their documentation below.

As shown above, you need only set the handler to the desired function to change behavior. This can be done either when
building the class definition, or during the __init__ call.

There are multiple utility functions defined under `bincfg.normalization.norm_utils` that can be used to set the handlers
above to different common behaviors without having to implement those functions yourself.

One may also set the `DEFAULT_TOKENIZATION_LEVEL` attribute on the class definition/instances to change what the default
tokenization level behavior will be.

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   bincfg.normalization.java
   bincfg.normalization.x86

Submodules
----------

bincfg.normalization.base\_normalizer module
--------------------------------------------

.. automodule:: bincfg.normalization.base_normalizer
   :members:
   :undoc-members:
   :show-inheritance:

bincfg.normalization.base\_tokenizer module
-------------------------------------------

.. automodule:: bincfg.normalization.base_tokenizer
   :members:
   :undoc-members:
   :show-inheritance:

bincfg.normalization.multi\_normalizer module
---------------------------------------------

.. automodule:: bincfg.normalization.multi_normalizer
   :members:
   :undoc-members:
   :show-inheritance:

bincfg.normalization.norm\_funcs module
---------------------------------------

.. automodule:: bincfg.normalization.norm_funcs
   :members:
   :undoc-members:
   :show-inheritance:

bincfg.normalization.norm\_utils module
---------------------------------------

.. automodule:: bincfg.normalization.norm_utils
   :members:
   :undoc-members:
   :show-inheritance:

bincfg.normalization.normalize module
-------------------------------------

.. automodule:: bincfg.normalization.normalize
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: bincfg.normalization
   :members:
   :undoc-members:
   :show-inheritance:
