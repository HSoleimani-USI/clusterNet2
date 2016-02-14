'''
Created on Apr 2, 2015

@author: tim
'''
import ctypes as ct
import logging
import numpy as np
funcs = ct.cdll.LoadLibrary('libClusterNet.so')

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
funcs.fto_pinned.restype = ct.POINTER(ct.c_float)
funcs.fT.restype = ct.POINTER(FloatMatrix)
funcs.ftranspose.restype = ct.c_void_p

funcs.fget_CPUBatchAllocator.restype = ct.c_void_p
funcs.fget_GPUBatchAllocator.restype = ct.c_void_p
funcs.falloc_next_batch.restype = ct.c_void_p
funcs.freplace_current_with_next_batch.restype = ct.c_void_p
funcs.fgetBatchX.restype = ct.POINTER(FloatMatrix)
funcs.fgetBatchY.restype = ct.POINTER(FloatMatrix)
funcs.fgetOffBatchX.restype = ct.POINTER(FloatMatrix)
funcs.fgetOffBatchY.restype = ct.POINTER(FloatMatrix)

#funcs.fget_neural_net.restype = ct.c_void_p
#funcs.ffit_neural_net.restype = ct.c_void_p

funcs.fprintmat.restype == ct.c_void_p

funcs.ffabs.restype = ct.c_void_p
funcs.flog.restype = ct.c_void_p
funcs.fsqrt.restype = ct.c_void_p
funcs.fpow.restype = ct.c_void_p

funcs.flogistic.restype = ct.c_void_p
funcs.flogistic_grad.restype = ct.c_void_p
funcs.frectified.restype = ct.c_void_p
funcs.frectified_grad.restype = ct.c_void_p

funcs.fcopy.restype = ct.c_void_p

funcs.fadd.restype = ct.c_void_p
funcs.fsub.restype = ct.c_void_p
funcs.fmul.restype = ct.c_void_p
funcs.fdiv.restype = ct.c_void_p

funcs.feq.restype = ct.c_void_p
funcs.flt.restype = ct.c_void_p
funcs.fgt.restype = ct.c_void_p
funcs.fle.restype = ct.c_void_p
funcs.fge.restype = ct.c_void_p
funcs.fne.restype = ct.c_void_p

funcs.fsquared_diff.restype = ct.c_void_p


funcs.fvadd.restype = ct.c_void_p
funcs.fvsub.restype = ct.c_void_p
funcs.ftmatrix.restype = ct.c_void_p


funcs.frand.restype = ct.POINTER(FloatMatrix)
funcs.frandn.restype = ct.POINTER(FloatMatrix)
funcs.fsetRandomState.restype = ct.c_void_p

funcs.fget_clusterNet.restype = ct.c_void_p

funcs.fdot.restype = ct.c_void_p

funcs.fslice.restype = ct.c_void_p

funcs.fsoftmax.restype = ct.c_void_p
funcs.fargmax.restype = ct.c_void_p


pt_clusterNet = funcs.fget_clusterNet()


funcs.frowMax.restype = ct.c_void_p
funcs.frowSum.restype = ct.c_void_p
funcs.frowMean.restype = ct.c_void_p

funcs.fcolMax.restype = ct.c_void_p
funcs.fcolSum.restype = ct.c_void_p
funcs.fcolMean.restype = ct.c_void_p

funcs.ffmax.restype = ct.c_float
funcs.ffsum.restype = ct.c_float
funcs.ffmean.restype = ct.c_float


funcs.fget_Timer.restype = ct.c_void_p
funcs.ftick.restype = ct.c_void_p
funcs.ftock.restype = ct.c_float

funcs.ffree.restype = ct.c_void_p

funcs.fsortbykey.restype = ct.c_void_p

funcs.fscalar_mul.restype = ct.c_void_p

funcs.fget_view.restype = ct.c_void_p





    

