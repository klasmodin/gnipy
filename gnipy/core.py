#!/usr/bin/env python
# encoding: utf-8
"""
Foundation classes for the GNIPY library.

GNIPY is available under GNU GPL v3 license.
Documentation guidelines are available `here <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.

Created by Klas Modin on 2014-05-12.
"""

from numpy import array, dot, ones, zeros, hstack, asarray, ndarray
from numpy import diag, triu, isscalar, linspace, shape


class Solver(object):	
	"""
	Top-level abstract class for GNIPY solvers.

	Every GNIPY integration algorithm is of this type.
	Direct subclasses of :class:`Solver` are used when
	wrapping existing ODE codes. Normally :class:`Integrator` 
	is subclassed when implementing a new solver.
	"""
	
	def run(self, totaltime, stepsize, state, *args, **kwargs):
		raise NotImplementedError(':meth:`integrate` method for class %s is not implemented.'%str(type(self)))

class Integrator(Solver):
	"""
	Numerical flow map :math:`\Phi` of the form :math:`y_{k+1} = \Phi(h,y_{k})`, 
	where :math:`h` is the step size and :math:`y_k` is the state variable.
	"""

	def run(self, totaltime, stepsize, state, *args, **kwargs):
		"""
		Carry out the integration process.

		A generator is returned that succesively uses the
			self.__call__(self, stepsize, state, *args, **kwargs) 
		method, yielding the succesive states (as reference, not copy).

		Parameters
		----------
		totaltime : float
			Total integration time.
		state : array_like
			Initial state array (can have any shape).
		stepsize : float
			Time increment between steps.
		*args : list
			Arbitrary number of arguments to be passed too self.__call__.
		**kwargs : dictionary
			Arbitrary number of keyword arguments to be passed to self.__call__.

		Returns
		-------
		Python generator

		See also
		--------
		N/A

		Examples
		--------
		N/A
		"""
		nsteps = int(round(float(totaltime)/stepsize))

		# Main integration loop
		for k in xrange(nsteps):
			state = self.__call__(stepsize,state)
			yield state


		# if isscalar(yinit):
		# 	y = asarray(yinit)
		# else:
		# 	y = array(yinit)
		# nout = int(round(float(tspan)/stepsize))
		# tout = linspace(0,tspan,nout)
		# yout = zeros(hstack((nout,shape(y))),dtype=y.dtype)
		# yout[0] = y

		# # Main integration loop
		# for i in xrange(1,nout):
		# 	# self.__call__(stepsize,y)
		# 	y = self.__call__(stepsize,y)
		# 	yout[i] = y
		# 	# yout[i] = self.__call__(stepsize,yout[i-1])
		
		# return (tout,yout)
	
	def __call__(self, stepsize, state, *args, **kwargs):
		"""
		Evaluation of the numerical flow function.

		Take a step with step size 'h' from input phase data 'y'
		and return new output phase vector.
		The variable 'y' is typically overwritten (for efficiency).
		"""
		raise NotImplementedError(':meth:`__call__` method for class %s is not defined.'%str(type(self)))
	
	def __mul__(self,other):
		"""
		Return composition of two methods.
		
		Parameters
		----------
		other : Integrator
			The method that self will be composed with.
			Returning method is: (h,y) -> self(h,.) o other(h,y).
		"""
		if (not isinstance(other,Integrator)):
			raise AttributeError("Object %s must be a Integrator."%str(other))
		if (isinstance(self,Composition)):
			mclist_self = self.mclist
		else:
			mclist_self = [(self,1.0)]
		if (isinstance(other,Composition)):
			mclist_other = other.mclist
		else:
			mclist_other = [(other,1.0)]
		return Composition(mclist_self+mclist_other)
	
	def __pow__(self,coeff):
		"""
		Return method with "stepsize coefficient".
		
		Parameters
		----------
		coeff : scalar
			The coefficient. Returned method is (h,y) -> self(coeff*h,y).
		"""
		if (not isscalar(coeff)):
			raise AttributeError("Object %s must be a scalar."%str(coeff))
		return Composition([(self,coeff)])
	

class Composition(Integrator):
	"""
	Composition of two or more methods.
	"""
	name = "Composition"
	
	def __init__(self, mclist):
		"""
		Constructor for composition integrator.
		
		Parameters
		----------
		mlist : list of tuples (Integrator,coeff)
			List of tuples. Each tuple should contains a
			method, and the coefficient for that method.
		"""
		super(Composition, self).__init__()
		self.mclist = mclist
	
	def __call__(self, stepsize, state, *args, **kwargs):
		for (m,c) in self.mclist:
			state = m.__call__(c*stepsize, state, *args, **kwargs)
		return state
	
	def __pow__(self,coeff):
		if (not isscalar(coeff)):
			raise AttributeError("Object %s must be a scalar."%str(coeff))
		mclist_new = []
		for (m,c) in self.mclist:
			mclist_new += [(m,coeff*c)]
		return Composition(mclist_new)
	
	def __str__(self):
		namestr = ''
		for (m,c) in self.mclist:
			if c == 1.0:
				namestr += str(m)+' * '
			else:
				namestr += "%s**%s * "%(str(m),str(c))
		return namestr[:-3]
	


if __name__ == '__main__':
	pass
