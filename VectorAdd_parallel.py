#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 20:49:03 2018

@author: dawnstear
"""
# Other packages youll need: CUDAtoolkit and CUDAdriver
import numpy as np
from timeit import default_timer as timer
from numbapro import vectorize, cuda, float32  
# nvprof 
# One way to tell the compiler to generate an accelerated version 
# of our VectorAdd function is with a decorator:

# The first input parameter to this decorator is a list of strings containing the signature
# of the function to be accelerated. A function signature (or type signature, or method
# signature) defines input and output of functions or methods.

# Remember this function will be compiled to the gpu machine code and this compiler needs
# to know the data types of the input and output, lets assume float32 for this example
# decorator(["output datatype", "input datatypes"], target)

@vectorize(["float32(float32, float32)"], target='gpu')    # by defult vectorize() uses a single threaded cpu
def VectorAdd(a,b):                                        # target can also be cuda/parallel
      #  for i in xrange(a.size):
      #  c[i] = a[i] + b[i]
    return a+b            # no need to pass in c for parallelization
                          # now the numbapro compiler can apply the scalar function
                          # automatically across the numpy arrays on gpu
        
        
def main():
        N = 32000000
        
        A = np.ones(N,dtype=np.float32)  
        B = np.ones(N,dtype=np.float32)
        C = np.zeros(N,dtype=np.float32)
        
        start = timer()
        # VectorAdd(A,B,C)  # the only other step is to then change how we call VectorAdd
        C = VectorAdd(A,B)
        VectorAdd_time = timer() - start
        
        print("C[:5] = " + str(C[:5]))
        print("C[-:5] = " + str(C[-5:]))
        print("VectorAdd took %f seconds" % VectorAdd_time)
        
if __name__ == '__main__':
    main()
