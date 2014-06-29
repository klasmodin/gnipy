"""
GNIPY: Geometric Numerical Integration in Python
================================================

A Python library for time-stepping algorithms for differential equation,
with particular focus on geometric numerical integration algorithms.

The library is designed to be:

Flexible 
	The class design is general enough to handle all types of 
	time-stepping algorithms.

Simple
	Using the algorithms is easy and implementing new methods 
	is straightforward, without overhead.

Efficient
	The library has intrisic support for linking in compiled code.


Routine listings
----------------
N/A

See also
--------
Documentation guidelines `here <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.

References
----------
N/A

Examples
--------
N/A
"""

from core import Solver, Integrator


__version__ = '0.0.1'
