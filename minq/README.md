# minq

Minq is a proxy app that replicates the linear algebra
components of a plane-wave or real-space density functional theory
code. It is based on the inq code (https://gitlab.com/npneq/inq).

The parallel linear algebra operations are done using the Slate
parallel linear algebra library (https://icl.utk.edu/slate/). In the
future it will also support for HeFFTe (for parallel fast Fourier
transforms) and the ELPA/ELSI libraries (for parallel eigenvalue
solves).

The code is being developed by Xavier Andrade <xavier@llnl.gov>.
