# Assured Timing Detector (ATD)
Version 1.0

The Assured Timing Detector software provides an implementation in C++ of an adaptable, model-based system for monitoring a timing signal for anomalies versus a reference time source ensemble.

Since precision timing is relied upon by numerous industry and infrastructure sectors, such anomaly detection could be used to activate failover operations (e.g., use of an alternative clock).  This system is designed to be:
* **Model-based,** in that a model of the dominant noise sources and clock dynamics over time is used to define expected behavior,
* **Adaptive,** in that calibration parameters needed for the model can be automatically fine-tuned from original estimates, and
* **Customizable,** in that a heterogeneous ensemble of time sources (ranging from internal clocks to network-connected time sources) can be used to fit the requirements of a given application.

This software was developed with funding from the Science and Technology Directorate of the U.S. Department of Homeland Security and under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory.

A companion status display software package, ATD-SD, is also available for use with this software.

Third-Party Library Dependencies
----------------
ATD relies upon third-party libraries for certain functionality.  Third-party software packages required by or suitable for use with this software (within directories labeled "thirdparty") are subject to their respective license terms and conditions. While these packages are not included in the software distribution, manifest files are included to indicate the source and expected contents of the corresponding directories within the build tree.  These dependencies include:
* [Boost software libraries](https://www.boost.org/users/history/version_1_66_0.html) (version 1.66.0, under Boost Software License v. 1.0)
* [Eigen linear algebra library](https://eigen.tuxfamily.org) (version 3.3.4, utilizing components under Mozilla Public License v. 2.0)
* [Date library](https://github.com/HowardHinnant/date/tree/master/include/date) (commit 2b6ee63 for date.h, under the MIT license)

Contributing
----------------
At this time, no contributions to ATD are being solicited from external contributors.

License
----------------
ATD is distributed under the terms of the MIT license.  See the LICENSE and NOTICE files in the root directory of this repository for details.

SPDX-License-Identifier: MIT

LLNL-CODE-837074
