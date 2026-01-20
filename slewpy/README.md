# slewpy

[![Documentation Status](https://readthedocs.org/projects/slewpy/badge/?version=latest)](https://slewpy.readthedocs.io/en/latest/?badge=latest)

slewpy is a Python package that allows the simulation of the science operations of an astrophysics space satellite mission. slewpy allows users to specify an astronomical target list with observing priorities and a satellite configuration (i.e., orbit and various satellite parameters). Taking these inputs, slewpy can be used to run a time-resolved simulation of an astrophysics mission and outputs simulated target observations given constraints such as satellite slewing rates between targets, observing time on a given target, and Sun-, Earth-, and Moon-limb observing constraints. slewpy provides the tools to test astrophysics space satellite mission designs against observing requirements for a given science case.

For more details on the project including the installation, contributing, and the getting started guide see the [documentation](https://slewpy.readthedocs.io/en/latest/).

![image info](./docs/source/slewpy_sim.gif)

Example visualization of a slewpy satellite simulation. The figure shows astrophysical transients appearing and disappearing on the celestial sphere (colored circles). Every time an unfilled circle appears around a transient, it indicates that the transient was observed by a satellite in a 600 km altitude sun-synchronous polar orbit. Shaded regions show various observing constraints: Moon, Sun, and Earth-limb constraints, as well as the high source density region of the Galactic plane. Transients are not observed when they are in one of the constrained regions.

## License

slewpy is distributed under the terms of the MIT license. All new contributions must be made under the MIT license.

See Link to [license](https://github.com/LLNL/slewpy/blob/main/LICENSE) and [NOTICE](https://github.com/LLNL/slewpy/blob/main/NOTICE) for details.

SPDX-License-Identifier: MIT

LLNL-CODE-2009734
