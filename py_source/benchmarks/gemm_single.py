'''
Created on Jan 3, 2016

@author: tim
'''
import cluster_net as gpu
import cudanet as gpu2



t = gpu.Timer()
dim1 = 128
dim_inner = 3072
dim_outer = 1000/96
'''
A = gpu.rand(dim1,dim_inner)
B = gpu.rand(dim_inner,dim_outer)
C = gpu.rand(dim1,dim_outer)


mean_time = 0
for i in range(5):
    iters = 100
    #warmup
    for j in range(1000):
        gpu.dot(A,B,C)
    t.tick(str(dim_inner))
    for j in range(iters):
        gpu.dot(A,B,C)
    ms = t.tock(str(dim_inner))
    mean_time+=ms/iters
    print "{0}ms".format(ms/iters)
    
print mean_time/5
'''

A2 = gpu2.random.rand(dim1,dim_inner)
B2 = gpu2.random.rand(dim_inner,dim_outer)
C2 = gpu2.random.rand(dim1,dim_outer)

mean_time = 0
for i in range(5):
    iters = 100
    #warmup
    for j in range(1000):
        gpu2.dot(A2,B2,C2)
    t.tick(str(dim_inner))
    for j in range(iters):
        gpu2.dot(A2,B2,C2)
    ms = t.tock(str(dim_inner))
    mean_time+=ms/iters
    print "{0}ms".format(ms/iters)
    
print mean_time/5
    
    
    
