'''
Created on Apr 2, 2015

@author: tim
'''
import ctypes as ct
import logging
import numpy as np
funcs = ct.cdll.LoadLibrary('libclusternet2.so')

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
funcs.fT.restype = ct.POINTER(FloatMatrix)

funcs.ffabs.restype = ct.c_void_p
funcs.flog.restype = ct.c_void_p
funcs.fsqrt.restype = ct.c_void_p
funcs.fpow.restype = ct.c_void_p


funcs.frand.restype = ct.POINTER(FloatMatrix)
funcs.frandn.restype = ct.POINTER(FloatMatrix)
funcs.fsetRandomState.restype = ct.c_void_p

funcs.fget_clusterNet.restype = ct.c_void_p

funcs.fdot.restype = ct.c_void_p


pt_clusterNet = funcs.fget_clusterNet()








    

