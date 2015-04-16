import numpy as np
from libcpp.string cimport string
from cpython cimport PyObject, Py_DECREF
from cpython cimport array
from cython.parallel import prange

cdef extern from "array_symbol.hpp":
	pass

cimport numpy as np
np.import_array()

cdef extern from "overfeat.hpp":
	cdef cppclass THTensor:
		pass

cdef extern from "overfeat.hpp" namespace "overfeat":
	cdef cppclass Overfeat:
		Overfeat(string, int, int) nogil
		THTensor* fprop(THTensor* input) nogil
		int get_n_layers() nogil
		THTensor* get_output(int) nogil

cdef extern from "of_util.hpp":
	cdef THTensor* THFromData(float* data, int, int, int)
	cdef THTensor* THFromContiguousArray(np.PyArrayObject*)
	cdef PyObject* ArrayFromTH(THTensor*)



cdef class PyOverfeat:
	cdef Overfeat* thisptr
	def __cinit__(self, filename, net_idx=1, max_layer=-1):
		self.thisptr = new Overfeat(filename, net_idx, max_layer)
	def __dealloc__(self):
		del self.thisptr
	def fprop(self, np.ndarray[float,ndim=3,mode="c"] img not None):
		#cdef np.ndarray input_c = np.PyArray_GETCONTIGUOUS(img)
		cdef THTensor* input_th = THFromContiguousArray(<np.PyArrayObject*>img)
		#print "in:", <long>input_th
		cdef THTensor* output_th = NULL
		with nogil:
				output_th = self.thisptr.fprop(input_th)
		#print "out:", <long>output_th
		output = ArrayFromTH(output_th)
		return <object>output
	def get_n_layers(self):
		return self.thisptr.get_n_layers()
	def get_output(self, int layer):
		cdef THTensor* output_th
		with nogil:
			output_th = self.thisptr.get_output(layer)
		output = ArrayFromTH(output_th)
		return <object>output

cdef class ParallelOverfeat:
	def __init__(self, filename, net_idx=1, max_layer=-1, num_threads=4):
		self.num_threads_ = num_threads
		assert(self.num_threads_ > 0)
		self.of_ = []
		for i in xrange(self.num_threads_):
			print "Initializing PyOverfeat worker {0}".format(i)
			self.of_[i] = PyOverfeat(filename, net_idx, max_layer)
			
	def get_n_layers(self):
		return self.of_[0].get_n_layers()
			
	def fprop(self, imgs):
		N = len(imgs)
#for i in prange(N, schedule='dynamic', nogil=False, num_threads=min(N,self.num_threads_)):
#		self.of_[i].fprop(imgs[i])
	
	def get_output(self, int layer, of_id):
		return self.of_[of_id].get_output(layer)
