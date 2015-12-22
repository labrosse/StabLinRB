[![Code Health](https://landscape.io/github/labrosse/StabLinRB/master/landscape.svg?style=flat)](https://landscape.io/github/labrosse/StabLinRB/master)

# StabLinRB
======

Computes the linear stability in the Rayleigh-Bénard problem.

The curve of critical Rayleigh number as function of wavenumber is computed
and plotted and the minimum value of the Rayleigh number is obtained along
with the corresponding value of the wavenumber.

Two choices of boundary conditions are possible: free-slip or no-slip, applying
to both boundaries. Cases with mixed BCs (Robin) or with different BCs at the
top and bottom still need to be implemented.

Options can be changed at the beginning of the file:
 - COMPUTE_FREESLIP and COMPUTE_NOSLIP are self-explaining boolean variables
 - NCHEB is the number of Chebyshev points to use in the computation
 - FTSZ is the fontsize for the annotation on the plots

The calculation uses an implementation of DMSuite in Python available on github
as part of the pyddx package (https://github.com/labrosse/pyddx).
DMSuite was originally developed for matlab by
Weidemann and Reddy and explained in ACM Transactions of Mathematical
Software, 4, 465-519 (2000). The present code is based on an octave code
originally developed by T. Alboussière and uses the Chebyshev differentiation
matrices.
