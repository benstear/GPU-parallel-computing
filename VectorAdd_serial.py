#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 18:47:28 2018

@author: dawnstear
"""

# graph theory coding

#
#    serial vs paralell computing
#   parallelizable or not? searching--yes, have 10 different workers each search a line
#find nth fibonacci number? no, each nth number depends on the previous number 


# faster code always means more efficient?

# if using cuda, file extension is .cu ,, nvcc = nvidia's compiler
# cuda architecture consists of threads, blocks and grids

# how does quadcore handle the problem of gpu computing, wherefibonacci has trouble??

# gpu invented bc each pixel can be computed independently, for transformations
# gpu's can have thousands of cores
# cpus = general, very fast , gpus = high bandwidth
# machine learning/DL also uses highly parallel massive matrix ops

# AWS is amazons cloud computing platform, google cloud is cheaper though


import numpy as np
from timeit import default_timer as timer

def VectorAdd(a,b,c):
    for i in xrange(a.size):
        c[i] = a[i] + b[i]
        
        
def main():
        N = 32000000
        
        A = np.ones(N,dtype=np.float32)  
        B = np.ones(N,dtype=np.float32)
        C = np.zeros(N,dtype=np.float32)
        
        
        start = timer()
        VectorAdd(A,B,C)
        VectorAdd_time = timer() - start
        
        print("C[:5] = " + str(C[:5]))
        print("C[-:5] = " + str(C[-5:]))
        print("VectorAdd took %f seconds" % VectorAdd_time)
        
if __name__ == '__main__':
    main()

        
        
        