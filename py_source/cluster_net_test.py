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
    
    A = np.float32(np.random.rand(100,1))    
    B = gpu.array(A.T)
    C = B.tocpu()    
    t.assert_array_equal(A.T,C,"Transpose error!")  
    
    A = np.float32(np.random.rand(1,100))    
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
    
    
def test_elementwise():
    A1 = np.float32(np.random.randn(100,100))
    A2 = np.float32(np.random.randn(100,100))    
    B1 = gpu.array(A1)
    B2 = gpu.array(A2)
    
    t.assert_almost_equal(gpu.abs(B1).tocpu(), np.abs(A1), 3, "Abs not working")
    t.assert_almost_equal(gpu.log(B1).tocpu(), np.log(A1), 3, "Log not working")
    t.assert_almost_equal(gpu.sqrt(B1).tocpu(), np.sqrt(A1), 3, "Sqrt not working")
    t.assert_almost_equal(gpu.pow(B1,2.0).tocpu(), np.power(A1,2.0), 3, "Pow not working")
    t.assert_almost_equal(gpu.logistic(B1).tocpu(), 1.0/(1.0+np.exp(A1)), 3, "Logistic not working")
    t.assert_almost_equal(gpu.logistic_grad(B1).tocpu(), A1*(A1-1.0), 3, "Logistic grad not working")
    t.assert_almost_equal(gpu.rectified_linear(B1).tocpu(), A1*(A1>0), 3, "Rectified not working")
    t.assert_almost_equal(gpu.rectified_linear_grad(B1).tocpu(), (A1>0), 3, "Rectified grad not working")
    
    t.assert_almost_equal(gpu.add(B1,B2).tocpu(), A1+A2, 3, "Add not working")
    t.assert_almost_equal(gpu.sub(B1,B2).tocpu(), A1-A2, 3, "Add not working")
    t.assert_almost_equal(gpu.mul(B1,B2).tocpu(), A1*A2, 3, "Add not working")
    t.assert_almost_equal(gpu.div(B1,B2).tocpu(), A1/A2, 3, "Add not working")
    
    t.assert_almost_equal(gpu.equal(B1,B2).tocpu(), np.equal(A1,A2), 3, "Add not working")
    t.assert_almost_equal(gpu.less(B1,B2).tocpu(), np.less(A1,A2), 3, "Add not working")
    t.assert_almost_equal(gpu.greater(B1,B2).tocpu(), np.greater(A1,A2), 3, "Add not working")
    t.assert_almost_equal(gpu.greater_equal(B1,B2).tocpu(), np.greater_equal(A1,A2), 3, "Add not working")
    t.assert_almost_equal(gpu.less_equal(B1,B2).tocpu(), np.less_equal(A1,A2), 3, "Add not working")
    t.assert_almost_equal(gpu.not_equal(B1,B2).tocpu(), np.not_equal(A1,A2), 3, "Add not working")
    t.assert_almost_equal(gpu.squared_difference(B1,B2).tocpu(), (A1-A2)**2, 3, "Add not working")

    
def test_vectorwise():   
    A = np.float32(np.random.randn(100,100))
    v = np.float32(np.random.randn(1,100))    
    B = gpu.array(A)
    V = gpu.array(v)
    
    t.assert_almost_equal(gpu.vector_add(B, V).tocpu(), A+v, 3, "Vec add not working")
    
def slice_test():
    A = np.float32(np.random.randn(100,100))
    B = gpu.array(A)
    C = gpu.slice(B,17,83,7,23).tocpu()
    print A
    
    t.assert_almost_equal(C, A[17:83,7:23], 3, "Slicing not working")
