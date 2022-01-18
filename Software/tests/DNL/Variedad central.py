#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 19:32:23 2019

@author: matias
"""
import numpy as np
import sympy as sp
from numpy import linalg as LA
from numpy.linalg import inv


a = np.matrix([[1,4,0,1,0],[0,4,0,0,0],[0,0,-4,0,0],[0,0,0,-1,0],[0,0,0,0,0]])

lamda, T = LA.eig(a)
x,y,v,w,gamma = sp.symbols('x,y,z,w,gamma')
a,b,c,d,e = sp.symbols('a,b,c,d,e')

X = np.array([x,y,v,w,gamma]).reshape(5,1)
U = np.array([a,b,c,d,e]).reshape(5,1)

X = T * U 

x = X[0,0]
y = X[1,0]
v = X[2,0]
w = X[3,0]
gamma = X[4,0]

f1 = x**2+v*x - 2*v
f2 = x*y-2*y*v-v*x*gamma
f3 = -v*x*gamma+2*v**2
f4 = x*w+2*v*w
f5=0

E = np.eye(len(lamda))*lamda




#%%