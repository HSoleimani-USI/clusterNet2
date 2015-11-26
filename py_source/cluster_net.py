import library_interface as lib
import numpy as np
import ctypes as ct


class array(object):
	def __init__(self, numpy_array=None, pt=None, shape=None):
		if pt != None: self.pt = pt
		if shape != None: self.shape = shape		
		if numpy_array != None:
			if len(numpy_array.shape) > 2: raise Exception("Array must be one or two dimensional!")
			self.shape = numpy_array.shape
			if len(self.shape) == 1:
				self.pt = lib.funcs.fempty(numpy_array.shape[0], 1)
			else:
				self.pt = lib.funcs.fempty(numpy_array.shape[0], numpy_array.shape[1])
			lib.funcs.fto_gpu(numpy_array.ctypes.data_as(ct.POINTER(ct.c_float)), self.pt)
		
		self.cpu_arr = numpy_array
		


	def tocpu(self):
		if self.cpu_arr == None: self.cpu_arr = np.empty(self.shape, dtype=np.float32)
		lib.funcs.fto_host(self.pt,self.cpu_arr.ctypes.data_as(ct.POINTER(ct.c_float)))
		return self.cpu_arr
	
	@property
	def T(self): return array(None, self.fT(self.pt), self.shape[::-1])

	
def ones(shape, dtype=np.float32):
	rows, cols = handle_shape(shape)
	return array(None, lib.funcs.ffill_matrix(rows,cols,ct.c_float(1.0)), shape)


def setRandomState(seed): lib.funcs.fsetRandomState(lib.pt_clusterNet, seed)
def rand(rows, cols, dtype=np.float32):	return array(None, lib.funcs.frand(lib.pt_clusterNet,rows,cols),(rows, cols))
def randn(rows, cols, dtype=np.float32): return array(None, lib.funcs.frandn(lib.pt_clusterNet,rows,cols),(rows, cols))


def handle_shape(shape):
	if len(shape) == 1: return (shape[0],1)
	elif len(shape) > 2: raise Exception("Array must be one or two dimensional!")
	else: return shape
	
def dot(A,B,out=None):
	if not out: out = ones((A.shape[0],B.shape[1]))
	print A.shape, B.shape, out.shape
	lib.funcs.fdot(lib.pt_clusterNet, A.pt, B.pt, out.pt)
	return out
