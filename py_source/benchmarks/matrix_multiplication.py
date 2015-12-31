'''
Created on Dec 31, 2015

@author: tim
'''
import cluster_net as gpu
import time

t = gpu.Timer()
dim = 32
for i in range(1000):
    dim += 32
    A = gpu.rand(dim,dim)
    B = gpu.rand(dim,dim)
    C = gpu.rand(dim,dim)
    
    if dim > 0: iters = 1000
    if dim > 100: iters = 100
    if dim > 1000: iters = 10
    if dim > 3000: iters = 4
    
    if i < 10:
        #warmup
        for j in range(10000):
            gpu.dot(A,B,C)
    t.tick(str(dim))
    for j in range(iters):
        gpu.dot(A,B,C)
    sec = t.tock(str(dim))/1000.
    tilesA = (dim/16)*((dim/64) + (1 if dim % 64 > 0 else 0))
    tilesB = ((dim/64) + (1 if dim % 64 > 0 else 0))*((dim/16)*((dim/64) + (1 if dim % 64 > 0 else 0)))
    memops = (tilesA+tilesB)*16*64 + (dim**2)
    
    print sec / (memops*iters)    
    print (memops/sec)*4*(1024**-3)*iters
    
    
