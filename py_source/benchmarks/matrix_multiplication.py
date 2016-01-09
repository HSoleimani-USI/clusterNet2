'''
Created on Dec 31, 2015

@author: tim
'''
import cluster_net as gpu
import time

t = gpu.Timer()
dim1 = 128
dim_inner = 32
dim_outer = 256
for i in range(1000):
    dim_inner += 32
    A = gpu.rand(dim1,dim_inner)
    B = gpu.rand(dim_inner,dim_outer)
    C = gpu.rand(dim1,dim_outer)
    
    if dim_inner > 0: iters = 1000
    if dim_inner > 100: iters = 100
    if dim_inner > 1000: iters = 10
    if dim_inner > 3000: iters = 4
    
    #warmup
    for j in range(2):
        gpu.dot(A,B,C)
    t.tick(str(dim_inner))
    for j in range(iters):
        gpu.dot(A,B,C)
    sec = t.tock(str(dim_inner))/1000.
    tilesA = (dim1/16)*((dim_inner/64) + (1 if dim_inner % 64 > 0 else 0))
    tilesB = ((dim_inner/64) + (1 if dim_inner % 64 > 0 else 0))*((dim_inner/16)*((dim_outer/64) + (1 if dim_outer % 64 > 0 else 0)))
    memops = (tilesA+tilesB)*16*64 + (dim_inner*dim_outer)
    
    #print sec / (memops*iters)    
    #print (memops/sec)*4*(1024**-3)*iters
    #print iters*(dim**3)/(sec*1000*1000*1000)
    #print iters*(dim1*dim_inner*dim_outer)/(sec*1000*1000*1000)
    print iters*dim1*dim_inner*dim_outer/(6144.*1000*1000*1000)*24, sec
    
    A2 = gpu.rand(dim1,dim_inner)
    C2 = gpu.rand(dim1,dim_inner)
    v = gpu.rand(dim1,1)
    t.tick("add " + str(dim_inner))
    for j in range(iters):
        #gpu.add(A,A2, C2)
        #gpu.vector_add(A, v, C2)
        gpu.softmax(A, A2)
        
    sec = t.tock("add " + str(dim_inner))/1000.
    
    
    
