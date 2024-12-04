'''
In this file we make the plots of the Markov chains, the corner plots of the free parameters of the model and report the confidence intervals.
All this information will be stored at the output directory (called '/results/').
'''


import numpy as np; #np.random.seed(42)
import emcee
from matplotlib import pyplot as plt
import yaml

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_global = os.path.dirname(path_git)

os.chdir(path_git); os.sys.path.append('./fr_mcmc/')
from utils.plotter import Plotter
from utils.derived_parameters import derived_parameters
from config import cfg as config

os.chdir(path_git + '/fr_mcmc/plotting/')

def parameters_labels(index, model):
    if model == 'LCDM':
        if index == 4:
            return ['$M_{abs}$', '$r_{d}$', r'$\Omega_m$', '$H_{0}$']
        elif index == 31:
            return ['$M_{abs}$', r'$\Omega_m$', '$H_{0}$']
        elif index == 32:
            return ['$r_{d}$', r'$\Omega_m$', '$H_{0}$']
        elif index == 21:
            return [r'$\Omega_m$', '$H_{0}$']

    elif (model == 'HS' or model == 'ST' or model == 'EXP'):
        if index == 5:
            return ['$M_{abs}$', '$r_{d}$', '$\Omega_{m}^{f(R)}$', r'$b$', '$H_{0}^{f(R)}$']
        elif index == 41:
            return ['$M_{abs}$', '$r_{d}$', r'$b$', '$H_{0}^{f(R)}$']
        elif index == 42:
            return ['$M_{abs}$', '$\Omega_{m}^{f(R)}$', r'$b$', '$H_{0}^{f(R)}$']
        elif index == 43:
            return ['$r_{d}$', '$\Omega_{m}^{f(R)}$', r'$b$', '$H_{0}^{f(R)}$']
        elif index == 31:
            return ['$M_{abs}$', r'$b$', '$H_{0}^{f(R)}$']
        elif index == 32:
            return ['$r_{d}$', r'$b$', '$H_{0}^{f(R)}$']
        elif index == 33:
            return [r'$\Omega_m$^{f(R)}', r'$b$', '$H_{0}^{f(R)}$']

def run(filename):
    model = config.MODEL
    output_dir = config.OUTPUT_DIR
    index = config.LOG_LIKELIHOOD_INDEX

    output_path = path_global + output_dir + filename
    os.chdir(output_path)

    parameters_label = parameters_labels(index, model)
    if model == 'LCDM':
        reader = emcee.backends.HDFBackend(filename + '.h5')
        samples = reader.get_chain()
        burnin = int(0.2*len(samples[:,0])); thin=1
        analysis = Plotter(reader, parameters_label, 'Title')

    else:    
        reader = emcee.backends.HDFBackend(filename + '.h5')
        samples = reader.get_chain()
        burnin = int(0.2*len(samples[:,0])); thin=1
        samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
        new_samples = derived_parameters(reader,discard=burnin,thin=thin,model=model)
        np.savez(filename+'_deriv', new_samples=new_samples)


        with np.load(filename + '_deriv.npz') as data:
            ns = data['new_samples']
        analysis = Plotter(ns, parameters_label, '')
        burnin = 0 # already has the burnin
        thin = 1

    results_dir = '/results'
    if not os.path.exists(output_path + results_dir):
            os.mkdir(output_path + results_dir)
 
    analysis.plot_contours(discard=burnin, thin=thin)
    plt.savefig(output_path + results_dir + '/cornerplot.png')
    plt.close()

    #if model == 'LCDM':
    analysis.plot_chains()
    #else:
    #    analysis.plot_chains_derivs()
    plt.savefig(output_path + results_dir + '/chains.png')
    plt.close()
    analysis.report_intervals(discard=burnin, thin=thin, save_path = output_path + results_dir)

    textfile_witness = open(output_path + results_dir + '/metadata.dat','w')
    textfile_witness.write('{}'.format(config))

if __name__ == "__main__":
    run('sample_LCDM_SN_2params')


