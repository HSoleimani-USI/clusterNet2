'''
Created on Jan 3, 2016

@author: tim
'''
import cluster_net as gpu2
import cudanet as gpu
import time


rdm = gpu.random.randn

t = gpu2.Timer()

batch_size = 64
hidden_size = 256
input_cols = 128
T = 1000

w_stacking = 4

input = rdm(batch_size,input_cols)
output = rdm(batch_size,hidden_size)

W = rdm(input_cols,hidden_size*4)
output_stacked = rdm(batch_size,hidden_size*4)
R = rdm(hidden_size,hidden_size*w_stacking)

inputs_stackedW = rdm(input_cols , T*batch_size)
errors_stackedW = rdm(T*batch_size, hidden_size*4)

inputT = rdm(input_cols, batch_size)
outputT = rdm(hidden_size, batch_size)
errors_W = rdm(batch_size, hidden_size*4)
errors_R = rdm(batch_size, hidden_size*4)

inputs_stackedR = rdm(hidden_size, T*batch_size)
errors_stackedR = rdm(T*batch_size, hidden_size*w_stacking)

iters = 500
mean_time = 0
t0 = time.time()
t.tick("stacking")
for i in range(iters):
    #for step in range(T):
    #gpu.dot(inputs_stackedW,errors_stackedW, W)
    gpu.dot(inputs_stackedR,errors_stackedR, R)
    
    
print "{0}.ms".format(t.tock("stacking")/iters)
print (time.time()-t0)/iters*1000

'''
iters = 5
mean_time = 0
t0 = time.time()
for i in range(iters):
    t.tick("no stacking")
    for step in range(T):
        gpu.dot(inputT,errors_W, W)
        gpu.dot(outputT,errors_R, R)
    
    t.tick("no stacking")
    
print "{0}.ms".format(t.tock("no stacking")/iters)
print (time.time()-t0)/iters*1000
'''
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

'''
    