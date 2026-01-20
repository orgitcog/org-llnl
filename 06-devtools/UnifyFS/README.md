# UnifyFS: A User-Level File System for Supercomputers

Node-local storage is becoming an indispensable hardware resource on
large-scale supercomputers to buffer the bursty I/O from scientific
applications. However, there is a lack of software support for node-local storage to
be used efficiently by applications that use shared files.

UnifyFS is an ephemeral, user-level file system under active development.
UnifyFS addresses a major usability factor of current and future systems because it enables
applications to gain performance advantages from distributed storage devices on the system while being as easy to use as a center-wide parallel file system.

## Documentation
UnifyFS documentation is at [https://unifyfs.readthedocs.io](https://unifyfs.readthedocs.io).

For instructions on how to build and install UnifyFS,
see [Build UnifyFS](http://unifyfs.readthedocs.io/en/dev/build.html).

## Build Status
Status of UnifyFS development branch (dev):

![Build Status](https://github.com/LLNL/UnifyFS/actions/workflows/build-and-test.yml/badge.svg?branch=dev)

[![Read the Docs](https://readthedocs.org/projects/unifyfs/badge/?version=dev)](https://unifyfs.readthedocs.io)

## UnifyFS Citation
We recommend that you use this citation for UnifyFS:

  * Michael Brim, Adam Moody, Seung-Hwan Lim, Ross Miller, Swen Boehm, Cameron Stanavige, Kathryn Mohror, Sarp Oral, “UnifyFS: A User-level Shared File System for Unified Access to Distributed Local Storage,” 37th IEEE International Parallel & Distributed Processing Symposium (IPDPS 2023), St. Petersburg, FL, May 2023.

## Contribute and Develop
If you would like to help, please see our [contributing guidelines](https://unifyfs.readthedocs.io/en/dev/contribute-ways.html).
