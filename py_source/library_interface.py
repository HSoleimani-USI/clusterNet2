'''
Created on Apr 2, 2015

@author: tim
'''
import ctypes as ct
import logging
import numpy as np
funcs = ct.cdll.LoadLibrary('libfoo.so')

class FloatMatrix(ct.Structure):
    _fields_ = [('rows', ct.c_int),
                ('cols', ct.c_int),
                ('bytes', ct.c_size_t),
                ('size', ct.c_int),
                ('data', ct.POINTER(ct.c_float))]   
    def __init__(self): pass    


funcs.fempty.restype = ct.POINTER(FloatMatrix)
funcs.ffill_matrix.restype = ct.POINTER(FloatMatrix)
funcs.fto_host.restype = ct.c_void_p
funcs.fto_gpu.restype = ct.c_void_p


class lib(object): 
    funcs = funcs






    

