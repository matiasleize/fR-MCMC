#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 20:42:21 2019

@author: matias
"""

import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
from matplotlib import pyplot as plt

def dX_dz(z, variables, gamma=0): 

    x = variables[0]
    y = variables[1]
    v = variables[2]
    w = variables[3]
    
    s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
    s1 = - (v*x*gamma - x*y + 4*y - 2*y*v) / (z+1)
    s2 = -v * (x*gamma + 4 - 2*v) / (z+1)
    s3 = w * (-1 + x+ 2*v) / (z+1)
    s4 = 1
    
    return [s0,s1,s2,s3,s4]


gamma = 0
x,y,v,w,z = sp.symbols('x,y,v,w,z')

s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
s1 = - (v*x*gamma - x*y + 4*y - 2*y*v) / (z+1)
s2 = -v * (x*gamma + 4 - 2*v) / (z+1)
s3 = w * (-1 + x+ 2*v) / (z+1)
s4 = 1

s = np.array([s0,s1,s2,s3])
var = np.array([x,y,v,w,z]) 
#Puntos fijos
pf =  sp.solvers.solve((s0,s1,s2,s3), (x,y,v,w,z))
pf1 = [list(pf[i]) for i in range(len(pf))]
print('Puntos fijos: {}, para todo z'.format(pf1))

#%%
# Matriz diferencial
s = np.array([s0,s1,s2,s3,s4])
var = np.array([x,y,v,w,z]) 
DD = np.zeros([5,5],dtype='object');
for (i,j) in np.ndindex(np.shape(DD)):
    DD[i,j] = sp.diff(s[i],(var[j]));
#%%
#Matriz evaluada:
def evaluate_matrix(Dif_matrix,var,fix_point):
    '''Toma una matriz de derivadas y evaluua las expresiones 
    en un punto fijo. '''
    if isinstance(fix_point,list):
        Dif_matrix_evaluate = np.zeros(np.shape(Dif_matrix),dtype='object');
        for (i,j) in np.ndindex(np.shape(Dif_matrix)):
            aux = sp.lambdify(list(var), Dif_matrix[i,j]);    
            Dif_matrix_evaluate[i,j] = aux(*fix_point)     
    else:
        Dif_matrix_evaluate = np.zeros(np.shape(Dif_matrix));
        for (i,j) in np.ndindex(np.shape(Dif_matrix)):
            aux = sp.lambdify(var, Dif_matrix[i,j]);    
            Dif_matrix_evaluate[i,j] = aux(fix_point)
#        np.array(Dif_matrix_evaluate)
    return Dif_matrix_evaluate


#Evaluamos en algun punto fijo, sin especificar z.
DD1 = evaluate_matrix(DD,var,pf1[3])

#Evaluamos en un redshift en particular para tener una matriz solo de numeros
DD2 = evaluate_matrix(DD1,z,1000)
#%%Separemos la parte linealizada de la no linalizada
diag = DD2.diagonal()
M1 = np.eye(len(diag)) * diag
M2 = DD2 - M1

#%%
from numpy import linalg as LA

#Evaluamos en un redshift en particular para tener una matriz solo de numeros
DD2 = evaluate_matrix(DD1,z,1000)

w, v = LA.eig(DD2)
print (w,v)
#%%
j=1 #Elijo algún punto fijo
eps = 10**(-8)
sol = solve_ivp(dX_dz, [1000, 0], np.array(pf1[j])+eps) #Me alejo un poquito y veo la dinámica

plt.close()
f, axes = plt.subplots(2,2)
ax = axes.flatten()
color = ['b','r','g','y']
[ax[i].plot(sol.t,sol.y[i],color[i]) for i in range(4)]
print(pf1[j])
#%% Campo de velocidades (para v,x, y) 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

plt.close()
W=50

fig = plt.figure()
ax = fig.gca(projection='3d')
x, y, v = np.mgrid[-W:W:8j, -W:W:8j, -W:W:8j]

gamma = 1
s0 = (x**2 + (1+v)*x - 2*v + 4*y)
s1 = - (v*x*gamma - x*y + 4*y - 2*y*v)
s2 = -v * (x*gamma + 4 - 2*v)

ax.quiver(x, y, v, s0, s1, s2,length=0.005) #Falta labelear los ejes

plt.show()