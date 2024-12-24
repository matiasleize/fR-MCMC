#For fancy plots
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

from scipy.stats import chi2
from scipy.special import erfinv

# import libraries:
import getdist
from getdist import plots, MCSamples
getdist.chains.print_load_details = False

import numpy as np
import git
import sys, os
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'../../..')))
# import tension tools utilities:
from tensiometer import utilities


import numpy as np; #np.random.seed(42)
import emcee
from matplotlib import pyplot as plt
import yaml

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_global = os.path.dirname(path_git)

os.chdir(path_git); os.sys.path.append('./fr_mcmc/')
os.chdir(path_git + '/fr_mcmc/plotting/')
output_dir = '/fR-output/HS/'


filename='sample_HS_BAO_full_3params'
output_path = path_global + output_dir + filename
os.chdir(output_path)
with np.load(filename+'_deriv.npz') as data:
    sample_HS_BAO = data['new_samples']


filename='sample_HS_PPS_4params'
output_path = path_global + output_dir + filename
os.chdir(output_path)
with np.load(filename+'_deriv.npz') as data:
    sample_HS_PPS = data['new_samples']

filename='sample_HS_PPS_BAO_full_4params'
output_path = path_global + output_dir + filename
os.chdir(output_path)
with np.load(filename+'_deriv.npz') as data:
    sample_HS_PPS_BAO = data['new_samples']

#print(sample_HS_PPS_BAO)
#plt.hist(sample_HS_PPS_BAO[:,2])
#plt.show()

chain_A = MCSamples(samples=sample_HS_PPS,names=['M_{abs}','\\Omega_m','b','H_{0}'], labels=['M_{abs}','\\Omega_m','b','H_{0}'])
chain_B = MCSamples(samples=sample_HS_BAO,names=['\\Omega_m','b','H_{0}'], labels=['\\Omega_m','b','H_{0}'])
chain_AB = MCSamples(samples=sample_HS_PPS_BAO,names=['M_{abs}','\\Omega_m','b','H_{0}'], labels=['M_{abs}','\\Omega_m','b','H_{0}'])

# Analyze the chains
# Example: Generate a triangle plot
g = plots.get_subplot_plotter()
g.triangle_plot(chain_A)
g.triangle_plot(chain_B)
g.triangle_plot(chain_AB)

#plt.show()
# Save the plot
#g.export('triangle_plott.png')

param_names = ['\\Omega_m','b','H_{0}'] #Shared paameters

#Estimate the mean and covariance of the chains:
mean_A = chain_A.mean(param_names)
cov_A = chain_A.cov(param_names)
mean_B = chain_B.mean(param_names)
cov_B = chain_B.cov(param_names)
mean_AB = chain_AB.mean(param_names)
cov_AB = chain_AB.cov(param_names)

from tensiometer import gaussian_tension

gaussian_A = gaussian_tension.gaussian_approximation(chain_A)
gaussian_B = gaussian_tension.gaussian_approximation(chain_B)
gaussian_AB = gaussian_tension.gaussian_approximation(chain_AB)

mean_A = chain_A.mean(param_names)
cov_A = chain_A.cov(param_names)
mean_B = chain_B.mean(param_names)
cov_B = chain_B.cov(param_names)
mean_AB = chain_AB.mean(param_names)
cov_AB = chain_AB.cov(param_names)
print(mean_A), print(cov_A)
print(mean_B), print(cov_B)
print(mean_AB), print(cov_AB)

beta = np.dot((mean_A - mean_B).T, np.dot(np.linalg.inv(cov_A + cov_B), (mean_A - mean_B)))
print(r'beta: {}'.format(beta))

df=3
PTE = chi2.sf(beta, df)
print('PTE: {}'.format(PTE))

n_sigma = utilities.from_confidence_to_sigma(1-PTE)
print('n_sigma: {:5g}'.format(n_sigma))

g.triangle_plot([chain_A,gaussian_A], params=param_names, legend_labels=['Planck18','Planck18 Gaussian'], filled=True)
g.triangle_plot([chain_B,gaussian_B], params=param_names, legend_labels=['BAO','BAO Gaussian'], filled=True)
g.triangle_plot([chain_AB,gaussian_AB], params=param_names, legend_labels=['Planck18+BAO','Planck18+BAO Gaussian'], filled=True)
#plt.show()

# import the module:
from tensiometer import mcmc_tension

# set single thread for running on small machines:
mcmc_tension.n_threads = 1
#param_names = ['Omega_m','H0','omega_b']
# create the distribution of the parameter differences:

diff_chain = mcmc_tension.parameter_diff_chain(chain_A, chain_B, boost=1)
diff_chain.name_tag = 'MCMC difference'
param_names_diff = ['delta_\\Omega_m', 'delta_b','delta_H_{0}']
#print(diff_chain.numrows)
print(diff_chain)

shift_probability, shift_prob_low_1, shift_prob_hi_1 = mcmc_tension.kde_parameter_shift(diff_chain, feedback=0, param_names=param_names_diff)
print(f'Shift probability considering all parameters = {shift_probability:.5f} +{shift_prob_hi_1-shift_probability:.5f} -{shift_probability-shift_prob_low_1:.5f}')
    
# turn the result to effective number of sigmas:
print(f'    PTE = {(1-shift_probability):.3f}')
print(f'    n_sigma = {utilities.from_confidence_to_sigma(shift_probability):.3f}')


# plot the contour for a sanity check:
g = plots.get_single_plotter()
g.settings.num_plot_contours = 3

g.plot_2d(diff_chain, 'delta_\\Omega_m', 'delta_H_{0}', filled=True)

g.add_x_marker(0)
g.add_y_marker(0)
plt.show()
shift_probability, shift_prob_low_1, shift_prob_hi_1 = mcmc_tension.kde_parameter_shift(diff_chain, feedback=0, param_names=param_names_diff)
print(f'Shift probability considering all parameters = {shift_probability:.5f} +{shift_prob_hi_1-shift_probability:.5f} -{shift_probability-shift_prob_low_1:.5f}')
 