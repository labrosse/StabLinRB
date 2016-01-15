[![Code Health](https://landscape.io/github/labrosse/StabLinRB/master/landscape.svg?style=flat)](https://landscape.io/github/labrosse/StabLinRB/master)

# StabLinRB
======

Computes the linear stability in the Rayleigh-Bénard problem.

The curve of critical Rayleigh number as function of wavenumber is computed
and plotted and the minimum value of the Rayleigh number is obtained along
with the corresponding value of the wavenumber. The first unstable
mode can also be plotted, both as profiles of the z dependence and as
temperature-velocity maps.

Any type of the classical boundary conditions can be computed at
either boundary for each variable, horizontal velocity (u), vertical
velocity (w), and temperature (t). Example calculations are provided
and cna be turned on or off by setting the options at the beginning of
the file:
* COMPUTE_FREESLIP for free slip BCs at both boundaries
* COMPUTE_NOSLIP for rigid BCs at both boundaries
* COMPUTE_FREERIGID for free slip top BC and rigid bottom
Resulting figures are also provided. Results for other BCs can be
obtained following this last example.

Other options can be changed at the beginning of the file:
* NCHEB is the number of Chebyshev points to use in the computation
* FTSZ is the fontsize for the annotation on the plots

Example of figures obtained for the most classical cases of boundary
conditions (combinations of free-slip and rigid on either sides) are
provided in pdf format.

The calculation uses an implementation of DMSuite in Python available on github
as part of the pyddx package (https://github.com/labrosse/dmsuite).
DMSuite was originally developed for matlab by
Weidemann and Reddy and explained in ACM Transactions of Mathematical
Software, 4, 465-519 (2000). The present code is based on an octave code
originally developed by T. Alboussière and uses the Chebyshev differentiation
matrices.
