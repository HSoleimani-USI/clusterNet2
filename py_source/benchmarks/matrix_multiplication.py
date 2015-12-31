'''
Created on Dec 31, 2015

@author: tim
'''
import cluster_net as gpu
import time


dim = 1
for i in range(1000):
    dim += 32
    print dim
    A = gpu.rand(dim,dim)
    B = gpu.rand(dim,dim)
    C = gpu.rand(dim,dim)
    gpu.dot(A,B,C)
    time.sleep(0.1)
    
    
