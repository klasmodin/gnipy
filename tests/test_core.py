#!/usr/bin/env python
# encoding: utf-8
"""
Tests for foundation classes for the GNIPY library.

GNIPY is available under GNU GPL v3 license.
Documentation guidelines are available `here <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.

Created by Klas Modin on 2014-06-27.
"""

import numpy as np
import gnipy as gp
import itertools as it

# Define various integrators to be tested

class Identity(gp.Integrator):

	def __call__(self, stepsize, state):
		return state

class DahlquistExplicitEuler(gp.Integrator):

	def __init__(self, lam=-1.0):
		self.lam = lam

	def __call__(self, stepsize, state):
		state *= 1.0+stepsize*self.lam
		return state

class NonNumPyIntegrator(gp.Integrator):

	def __call__(self, stepsize, ((q,p),a)):
		q += stepsize*p
		p -= stepsize*q
		a -= stepsize
		return ((q,p),a)

class VerletA(gp.Integrator):

	def __call__(self, stepsize, (q,p)):
		q += stepsize*p
		return (q,p)

class VerletB(gp.Integrator):

	def __call__(self, stepsize, (q,p)):
		p -= stepsize*q
		return (q,p)

class StormerVerlet(gp.Integrator):
	def __call__(self, stepsize, (q,p)):
		half = stepsize/2.0
		q += half*p
		p -= stepsize*q
		q += half*p
		return (q,p)


# Define test class

class TestCore(object):
	identity = Identity()
	sol1 = DahlquistExplicitEuler()
	sol2 = DahlquistExplicitEuler(1.0j)
	sol3 = NonNumPyIntegrator()
	solA = VerletA()
	solB = VerletB()
	sol_sv = StormerVerlet()

	def test_identity(self):
		y = np.ones(1e6,dtype=float)
		y2 = y
		for y in self.identity.run(totaltime=1.0, stepsize=1e-6, state=y): pass
		assert(y is y2)

	def test_large(self):
		y = np.ones(1e6,dtype=float)
		for y in self.sol1.run(totaltime=1.0, stepsize=1e-3, state=y): pass
		np.testing.assert_allclose(y[:2],np.array([0.36769542,0.36769542]),atol=1e-8,rtol=1e-8)

	def test_complex(self):
		y = np.ones(10,dtype=complex)
		for y in self.sol2.run(1.0, 1e-3, y): pass
		yabs = abs(y)
		np.testing.assert_allclose(yabs,1.,atol=1e-3)

	def test_scalar_dahlquist(self):
		y = 1.0
		for y in self.sol1.run(totaltime=1.0, stepsize=1e-3, state=y): pass
		np.testing.assert_allclose(y,0.36769542,atol=1e-8,rtol=1e-8)

	def test_nonnumpy(self):
		q = np.ones(3,dtype=float)
		p = np.zeros_like(q)
		a = 1
		for ((q,p),a) in self.sol3.run(totaltime=1.0, stepsize=1e-3, state=((q,p),a)): pass
		np.testing.assert_allclose(a,0.,atol=1e-15)
		r = np.sqrt(q**2+p**2)
		np.testing.assert_allclose(r,1.,atol=1e-3)

	def test_scalar_nonnumpy(self):
		q = 1
		p = 0
		a = 1
		gen = self.sol3.run(totaltime=1.0, stepsize=1e-3, state=((q,p),a))
		try:
			while True:
				gen.next()
		except StopIteration, e:
			pass

	def test_composition(self):
		sol = self.sol2**0.5*self.sol1*self.sol2**0.5
		y = np.ones(2,dtype=complex)
		for y in sol.run(1.0, 1e-3, y): pass
		yabs = abs(y)
		np.testing.assert_allclose(yabs,0.36778736,atol=1e-6)

	def test_verlet(self):
		q = np.ones(1e5,dtype=float)
		p = np.zeros_like(q)
		for (q,p) in self.sol_sv.run(totaltime=1.0, stepsize=1e-3, state=(q,p)): pass
		r = np.sqrt(q**2+p**2)
		np.testing.assert_allclose(r,1.,atol=1e-13)

	def test_verlet_composition(self):
		q = np.ones(3,dtype=float)
		p = np.zeros_like(q)
		sol_v = self.solA**0.5*self.solB*self.solA**0.5
		for (q,p) in sol_v.run(totaltime=1.0, stepsize=1e-3, state=(q,p)): pass
		r = np.sqrt(q**2+p**2)
		np.testing.assert_allclose(r,1.,atol=1e-13)
		for (q,p) in self.sol_sv.run(totaltime=1.0, stepsize=1e-3, state=(q,-p)): pass
		np.testing.assert_allclose(q,1.,atol=1e-12)
		np.testing.assert_allclose(p,0.,atol=1e-12)


