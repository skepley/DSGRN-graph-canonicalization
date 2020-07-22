#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:09:35 2019

@author: sk2011
"""

import DSGRN
from SKModule import *
import numpy as np
#import oapackage



def is_canonical(idx):
    '''check if indices are fully partitioned'''
    
    return all([len(i) == 1 for i in idx])

def apply_permutation(matrix, permutation):
    '''apply a permutation to both the rows and columns of a matrix'''
    
    matrix[permutation,:]
    return matrix[:,permutation][permutation,:]

#netString = 'AAR' + 'R0R' + '00A'
#netMatrix = string2matrix(netString)



n = 10
#netMatrix = np.random.randint(-1,1,[n,n])
idx = [np.array([i for i in range(n)])] # initial index


#printL(netMatrix)
#print(idx)




i = idx[0]
inDegree = np.array([sum(abs(k) for k in netMatrix[j]) for j in i])
#print(inDegree)

sortIdx = np.argsort(inDegree)
#print(sortIdx)

sortedInDegree = list(inDegree[sortIdx])
splitNodes = 1+np.arange(n-1)[np.diff(sortedInDegree) != 0]
blockInDegree = np.split(sortedInDegree,splitNodes)
idx1 = np.split(sortIdx, splitNodes)
netMatrix1 = netMatrix[sortIdx]
#print(splitNodes)
printL(idx1)
printL(netMatrix1)
# print(is_canonical(idx))