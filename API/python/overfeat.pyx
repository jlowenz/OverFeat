import numpy as np
cimport numpy as np
from libcpp.string cimport string
from cpython cimport PyObject
from cpython cimport array

cdef extern from "overfeat.hpp":
	cdef cppclass THTensor:
		pass

cdef extern from "overfeat.hpp" namespace "overfeat":
	cdef cppclass Overfeat:
		Overfeat(string, int, int)
		THTensor* fprop(THTensor* input)
		int get_n_layers()
		THTensor* get_output(int)

cdef extern from "of_util.hpp":
	cdef THTensor* THFromData(float* data, int, int, int)
	cdef THTensor* THFromContiguousArray(np.PyArrayObject*)
	cdef PyObject* ArrayFromTH(THTensor*)


cdef class PyOverfeat:
	cdef Overfeat* thisptr
	def __cinit__(self, filename, net_idx, max_layer=-1):
		self.thisptr = new Overfeat(filename, net_idx, max_layer)
	def __dealloc__(self):
		del self.thisptr
	def fprop(self, np.ndarray[float,ndim=3,mode="c"] img not None):
		cdef THTensor* input_th = THFromContiguousArray(<np.PyArrayObject*>img)
		cdef THTensor* output_th = self.thisptr.fprop(input_th)
		output = ArrayFromTH(output_th)
		return <object>output
	def get_n_layers(self):
		return self.thisptr.get_n_layers()
	def get_output(self, int layer):
		cdef THTensor* output_th = self.thisptr.get_output(layer)
		output = ArrayFromTH(output_th)
		return <object>output
