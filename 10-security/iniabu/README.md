**THIS PROJECT HAS BEEN REPLACED AND IS THUS ARCHIVED. PLEASE SEE [HERE](https://github.com/galactic-forensics/iniabu) FOR THE NEW, COMPLETELY RE-WRITTEN VERSION**

# iniabu
This package can be used to query the solar system elemental and
isotopic composition. It is based on various databases. Currently
available databases are
 * [Lodders et al. (2009)](https://doi.org/10.1007/978-3-540-88055-4_34)

## Installation
The package can be installed via pip

    pip install iniabu

If you want to upgrade to the newest version, type:

    pip install iniabu --upgrade

Alternatively, less preferred (should only be used if, for some
reason, you cannot access pip), the software can be installed
by running:

    python setup.py

## Using the package
### Loading an instance
Once you have installed the class you can import it, as usual, in python
by calling

    import iniabu

You can now load your instance, e.g., as

    ini = iniabu.IniAbu()

Optionally, you can add an *fname* argument in this call and
select the database you would like to work with. By default
*fname='lodders09'* is loaded. Optionally, you can load separate
instances with different databases.

### A simple example
In order to calculate a delta value, you need to define
two isotopes that you calculate your delta value for your
measurement, e.g., you measured the isotopes <sup>29</sup>Si
and <sup>28</sup>Si. The standard format
to give the program an isotope is by passing a string, e.g.,
'Si-28' for <sup>28</sup>Si. The required delta value can then
be found by calling:

    measurement_value = 0.05
    ini.delta_iso(measurement_value, 'Si-29', 'Si-28')

Here, measurement_value is the measured value for which a
delta value comparing it to the Solar System average composition
will be calculated. The program then returns the delta value in
per mille:

    -14.957264957264904

More details can be found in the introduction of the docustring
to the class. If you're using ipython you can call it up via

    ini?

### Nomenclature & Options
**Isotopes** should be passed to the program in the form 'Fe56',
where 'Fe' is the symbol for the given element and '56' is the
total number of nucleons.

**massf**: Some routines have a *massf* argument. By default
this is set to *False*. In this scenario, it is assumed that your
isotope ratios are number ratios. If *massf = True*, the program
assumes for this calculation that your value (and the return,
where applicable) are in mass fractions.

**mult**: Functions calculating delta values usually have a
*mult* option. Delta values by themselves are not defined in
terms of per mille or any other fraction. The multiplier, which
is by default set to 1000 to return per mille, can be set to
return anything else as well, e.g., *mult=100* would return
delta values in percent.

### Available subroutines

The subroutines and data listed here are given by *name*. To
the function, type:

    ini.name(...)

and give the required arguments (if any). Docstrings are
available for all functions and can, from ipython, be called
by:

    ini.name?

Most users will want to use the functions for ratios.

#### Queries:

* **query_ele** returns the information that is stored on
a given element.
* **query_iso** returns the information that is stored on a
given isotope.

#### Functions for ratios:

* **delta_iso** returns the delta value of two given isotopes
* **delta_ele** returns the delta value of two given elements
* **bracket_iso** returns the bracket notation of a given pair of isotope
* **bracket_ele** returns the bracket notation of a given pair of elements

#### General information

* **data** holds all the read in information.
* **aa, ele, zz** holds the information for every entry in the
database for the total number of nucleons (*aa*), the total
number of protons (*zz*), and the element abbrevition (*ele*).
* **atomp** holds the information on atom percentage, i.e.,
how many percent a given isotope makes up of an element.
* **nn** holds the information on abundnace fraction of a given
isotope with respect to the average solar system.

### More information
More detailed information can be formed in the docstring of the
python class file. If you are using ipython, you can query
individual commands, e.g., as:

    ini.delta_iso?

## Contact
Feel free to contact me if you find a bug and would like to have
it fixed. You can find my e-mail address below. 

Please also let me know if you would like to have additional
functionality added. I don't expect you to contribute directly 
to this code, but please feel free to do so and create a new pull
request. Testing is all done manually at this point and I expect
it it remain like that for the foreseeable future. Good luck :)

## Release

LLNL-CODE-799977  

Copyright (c) 2019, Lawrence Livermore National Security,
LLC. Produced at the Lawrence Livermore National Laboratory.  
Written by Reto Trappitsch  
trappitsch1@llnl.gov

All rights reserved.

Please also read this link – Our Disclaimer and GNU General
Public License.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
(as published by the Free Software Foundation) version 2,
dated June 1991.

This program is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the IMPLIED
WARRANTY OF MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the terms and conditions of the GNU General
Public License for more details.

You should have received a copy of the GNU General Public
License along with this program; if not, write to the Free
Software Foundation, Inc., 59 Temple Place, Suite 330,
Boston, MA 02111-1307 USA
