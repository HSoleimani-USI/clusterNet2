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

    
