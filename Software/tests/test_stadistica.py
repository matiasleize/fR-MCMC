#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 23:03:17 2019

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
import math
from IPython.display import display, Math

#%%
Gauss = np.random.normal(10,2,10**5)
chi =1* Gauss**(2)

%matplotlib qt5
plt.close()
plt.figure()
plt.xlabel(r'$\beta$')
#plt.hist(Gauss,density=True,bins=round(np.sqrt(len(Gauss))),label=r'$Gauss$')
plt.hist(chi,density=True,bins=round(np.sqrt(len(chi))),label=r'$Gauss$')
plt.grid(True)
plt.legend()
#%%
txt = "\mathrm{{{2}}} = {0:.3f}\pm{{{1:.3f}}}"
txt = txt.format(np.mean(beta_posta), np.std(beta_posta), r'\beta')
display(Math(txt))
