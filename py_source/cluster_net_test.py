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
    
    A = np.float32(np.random.rand(5,7))
    B = gpu.array(A)
    C = B.tocpu()
    t.assert_array_equal(A, C, "To GPU does not work!")  
   
def test_togpu():
    A = np.float32(np.random.rand(17,83))
    B = gpu.array(A)
    C = B.tocpu()
    t.assert_array_almost_equal(A,C,4,"array.tocpu not equal to init array!")   
    
    
def test_dot():
    A1 = np.float32(np.random.rand(2,3))
    A2 = np.float32(np.random.rand(3,2))
    B1 = gpu.array(A1)
    B2 = gpu.array(A2)
    B3 = gpu.dot(B1,B2)
    C = B3.tocpu()
    
    t.assert_array_almost_equal(np.dot(A1,A2),C,4,"array.tocpu not equal to init array!")     
    
    
def test_Transpose():
    A = np.float32(np.random.rand(17,83))    
    B = gpu.array(A.T)
    
    
    C = B.tocpu()
    
    t.assert_array_equal(A.T,C,"Transpose error!")     
    
    
def test_rand():
    A = gpu.rand(100,100)
    B = A.tocpu()
    assert np.mean(B) > 0.45 and np.mean(B) < 0.55
    assert np.var(B) > 0.08 and np.var(B) < 0.085
    
    gpu.setRandomState(1234)
    A1 = gpu.rand(1000,1000)
    gpu.setRandomState(1234)
    A2 = gpu.rand(1000,1000)
    B1 = A1.tocpu()
    B2 = A2.tocpu()
    
    t.assert_array_equal(B1, B2, "Seed not working!")
    
    
def test_randn():
    A = gpu.randn(100,100)
    B = A.tocpu()
    assert np.mean(B) > -0.05 and np.mean(B) < 0.05
    assert np.var(B) > 0.95 and np.var(B) < 1.05
    
    gpu.setRandomState(1234)
    A1 = gpu.randn(1000,1000)
    gpu.setRandomState(1234)
    A2 = gpu.randn(1000,1000)
    B1 = A1.tocpu()
    B2 = A2.tocpu()
    
    t.assert_array_equal(B1, B2, "Seed not working!")  

    
