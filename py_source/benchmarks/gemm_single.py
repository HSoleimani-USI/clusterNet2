'''
outputreated on Jan 3, 2016

@author: tim
'''
import cluster_net as gpu
import cudanet as gpu2



t = gpu.Timer()
dim1 = 16
dim_inner = 1024
dim_outer = 1024

batch_first_mode = True

if batch_first_mode:
    input = gpu.rand(dim1,dim_inner)
    W = gpu.rand(dim_inner,dim_outer)
    output = gpu.rand(dim1,dim_outer)
    input2 = gpu2.random.rand(dim1,dim_inner)
    W2 = gpu2.random.rand(dim_inner,dim_outer)
    output2 = gpu2.random.rand(dim1,dim_outer)
else:    
    input = gpu.rand(dim_inner,dim1)
    W = gpu.rand(dim_outer,dim_inner)
    output = gpu.rand(dim_outer,dim1)
    
    input2 = gpu2.random.rand(dim_inner,dim1)
    W2 = gpu2.random.rand(dim_outer,dim_inner)
    output2 = gpu2.random.rand(dim_outer,dim1)


mean_time = 0
for i in range(5):
    iters = 100
    #warmup
    for j in range(1000):
        if batch_first_mode:
            gpu.dot(input,W,output)
        else:
            gpu.dot(W, input, output)
    t.tick(str(dim_inner))
    for j in range(iters):
        if batch_first_mode:
            gpu.dot(input,W,output)
        else:
            gpu.dot(W, input, output)
    t.tick(str(dim_inner))
    
print t.tock(str(dim_inner))/5/iters



mean_time = 0
for i in range(5):
    iters = 100
    #warmup
    for j in range(1000):
        if batch_first_mode:
            gpu2.dot(input2,W2,output2)
        else:
            gpu2.dot(W2,input2,output2)
    t.tick(str(dim_inner))
    for j in range(iters):
        if batch_first_mode:
            gpu2.dot(input2,W2,output2)
        else:
            gpu2.dot(W2,input2,output2)
    t.tick(str(dim_inner))
    
    
print t.tock(str(dim_inner))/5/iters
    
    
    

