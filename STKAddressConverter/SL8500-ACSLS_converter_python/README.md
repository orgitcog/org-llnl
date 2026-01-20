# acs2internal

This script translates an ACS / HLI-PRC address to its internal address or vice versa.
`acs2internal.py` is inside `acs2internal/` folder.


## Script
### Converting LTO5 HLI_PRC Address to Internal Address.

Example: 1,10,1,4 to 3,3,-1,1,1

Usage: `python acs2internal.py -d 1,10,1,4`

STDOUT: `3,3,-1,1,1`

### Converting LTO5 Internal Address to HLI-PRC Address.

Example: 3,3,-1,1,1 to 1,10,1,4 

Usage: `python acs2internal.py -i 3,3,-1,1,1`

STDOUT: `1,10,1,4`

`--debug` provides additional output.


## Pip library
### Building the wheel

Creating the .whl
```
python setup.py bdist_wheel
```

Creating the tar.gz
```
python setup.py sdist
```

### Installing the pip library from this folder

```
sudo pip install . --upgrade
```

### pip library usage

```
from acs2internal import acs2internal

internal_address = acs2internal.acsls_addr_to_internal_addr(acs_address="1,10,1,4")
print(internal_address)

> 3,3,-1,1,1
```

```
from acs2internal import acs2internal

acs_address = acs2internal.internal_addr_to_acsls_addr(internal_address="3,3,-1,1,1")
print(acs_address)

> 1,10,1,4
```
