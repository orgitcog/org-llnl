STK Address Converter
=====================

Overview
--------

Oracle StorageTek tape libraries have an addressing scheme that looks like:

    L,R,C,S,W
    L: Library
    R: Rail
    C: Column
    S: Side
    R: Row

The software that controls Oracle StorageTek tape libraries, ACSLS, has an
different addressing scheme that looks like:

    A,L,1,D
    A: ACS
    L: LSM
    D: Drive #

Even though ACSLS and StorageTek libraries work hand-in-hand, their drive
addressing and indentification schemes differ. For example,
a drive known by ACSLS as:

    1,4,1,4

is referred to by StorageTek tape libraries exclusively as:

    2,1,-1,1,1 

Converters
----------

A couple different flavors of converters are initially offered here. More may be
offered in the future as the need arises. Contributions of more converters are
encouraged.

For now, the converters offered are:

* *SL8500toACSLS.pm*: a Perl module that offers one-way conversion from the
StorageTek library addressing scheme to ACSLS' scheme
* *SL8500<->ACSLS_converter_python*: a Python module that can be imported or run as
a standalone program. It converts both ways between StorageTek library address
and ACSLS address schemes. This module is meant to be built into a wheel that
can be `pip install`'d.

Run the Python Version from the Command Line
--------------------------------------------

### Converting ACSLS Address to SL8500 Internal Address

Example: 1,10,1,4 to 3,3,-1,1,1

```
> python acs2internal.py -d 1,10,1,4
3,3,-1,1,1
```

### Converting SL8500 Internal Address to ACSLS Address

Example: 3,3,-1,1,1 to 1,10,1,4 

```
> python acs2internal.py -i 3,3,-1,1,1
1,10,1,4
```

`--debug` provides additional output.

Build the Python Package
------------------------

Creating the .whl

    python setup.py bdist_wheel

Creating the source dist package

    python setup.py sdist

Installing the module via pip

    pip install acs2internal

Module usage

```
from acs2internal import acs2internal

internal_address =
acs2internal.acsls_addr_to_internal_addr(acs_address="1,10,1,4")
print(internal_address)

> 3,3,-1,1,1
```

```
from acs2internal import acs2internal

acs_address =
acs2internal.internal_addr_to_acsls_addr(internal_address="3,3,-1,1,1")
print(acs_address)

> 1,10,1,4
```
