# WVL

The software runs using the Priithon image analysis platform (https://github.com/sebhaase/priithon), which is part of this release.
The Priithon manual and a tutorial can be downloaded from the github url.

The version of this mac distribution is:
Priithon_27_mac-20151127, which expands to a folder named Priithon_27_mac

The [paper](jmicrosc.pdf) is the reference source for the wavelet algorithm in this distribution; "A novel 3D wavelet-based filter for visualizing features in noisy biological data," J. Microsc. 219, 43-49 (2005).

This distribution has been tested on Mac OS 10.9.5, 10.10.3, and 10.11.1


## Installation

Copy the Priithon_27_mac-20151127.zip to your home directory (~), or the directory in which you want Priithon installed
unzip

## Execution

cd ~/Priithon_27_mac, or the path to /Priithon_27_mac
double click priithon_script, or in a Terminal window type ./priithon_script and press the return key

A priithon window will open.
To import the wavelet subroutines, type the following In the priithon window:
from Priithon import Wvl  (cap "W", lower case "v", lower case "ell")


## Wvl functions

Wvl.threeD : 3D wavelet, operates on a volume
Wvl.twoD : 2D wavelet, operates on a 2D array
Wvl.twoDt : 3D wavelet, operates on a time stack of 2D data; spatial and temporal wavelet sizes can differ
Wvl.oneDT : 1D wavelet, operates on a 1D array


## Notes

A wavelet of size "n" covers 3n voxels, so to ensure that larger wavelets can operate on a given data set, it may be necessary and often useful to use F.copyPadded to imbed the data into a larger area or volume. This larger array can be recopied into the original size after the wavelet operations are performed, to restore the original data extents.

## Example

Test.py is an example file that demonstrates the use of Wvl.twoD:

    start Priithon

type the following line in the priithon window and then press the return key

    fn=Y.FN()

navigate to the test.py file and select it. Type the following line in the priithon window and then press the return key

    execfile(fn)

The two blocks of output should be identical

## License

This code is released under a BSD license. For more details, see the [LICENSE] file.

``LLNL-CODE-696744``
