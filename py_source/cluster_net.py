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

	
def ones(shape, dtype=np.float32):
	rows, cols = handle_shape(shape)
	return array(None, lib.funcs.ffill_matrix(rows,cols,ct.c_float(1.0)), shape)

def handle_shape(shape):
	if len(shape) == 1: return (shape[0],1)
	elif len(shape) > 2: raise Exception("Array must be one or two dimensional!")
	else: return shape
