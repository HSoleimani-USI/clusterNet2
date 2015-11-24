import cluster_net as gpu
import nose
import numpy as np
import numpy.testing as t



def setup():
    pass

def teardown():
    pass

def test_cpu():  
    A = np.ones((5,7))
    B = gpu.ones((5,7))
    C = B.tocpu()
    t.assert_array_equal(A, C, "To GPU does not work!")  
   
def test_togpu():
    A = np.float32(np.random.rand(17,83))
    B = gpu.array(A)
    C = B.tocpu()
    t.assert_array_almost_equal(A,C,4,"array.tocpu not equal to init array!")   
    
    
def test_dot():
    A1 = np.float32(np.random.rand(2,2))
    A2 = np.float32(np.random.rand(2,2))
    B1 = gpu.array(A1)
    B2 = gpu.array(A2)
    B3 = gpu.dot(B1,B2)
    C = B3.tocpu()
    
    t.assert_array_almost_equal(np.dot(A1,A2),C,4,"array.tocpu not equal to init array!")  

    
