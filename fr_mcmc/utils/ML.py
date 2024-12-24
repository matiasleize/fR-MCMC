import numpy as np
import torch
from neurodiffeq.conditions import BundleIVP  # the initial condition
from neurodiffeq.solvers import BundleSolution1D
import torch
from ML_utils import CustomCondition, f_R_reparams, _shape_manager
import yaml
from scipy.constants import c as c_ms
from matplotlib import pyplot as plt

model = 'Hu-Sawicki'
path = f'{model}_nets'

c = c_ms/1000

#with open(f'../trained_net/{path}/saved_config.yaml') as yaml_file_f_R:
with open(f'{path}/saved_config.yaml') as yaml_file_f_R:
    try:
        Config_f_R = yaml.safe_load(yaml_file_f_R)
    except yaml.YAMLError as exc:
        print(exc)

#with open(f'../trained_net/{path}/saved_d_L_config.yaml') as yaml_file_f_R_d_L:
with open(f'{path}/saved_d_L_config.yaml') as yaml_file_f_R_d_L:
    try:
        Config_f_R_d_L = yaml.safe_load(yaml_file_f_R_d_L)
    except yaml.YAMLError as exc:
        print(exc)

# Set the range of the independent variable:
z_0 = 10.0
z_rescale = z_0
z_d_L_rescale = Config_f_R_d_L['z_rescale']

z_prime_min = 0.0
z_prime_max = 1.0

z_min = z_prime_min
z_max = min(z_rescale, z_d_L_rescale)*z_prime_max

# Set the range of the parameters of the bundle:
b_prime_min = float(Config_f_R['b_prime_min'])
b_prime_max = 1.0

b_max = Config_f_R['b_max']
b_min = b_prime_min*b_max

Om_m_0_min = Config_f_R['Om_m_0_min']
Om_m_0_max = Config_f_R['Om_m_0_max']

# Set neural network parameters and renormalizations:
alpha_nominator = Config_f_R['alpha_nominator']
alpha_denominator = Config_f_R['alpha_denominator']
alpha = alpha_nominator/alpha_denominator

nets = torch.load(f'../trained_net/{path}/nets_f_R.ph', map_location=torch.device('cpu'),weights_only=False)
net_INT = torch.load(f'../trained_net/{path}/nets_f_R_d_L.ph', map_location=torch.device('cpu'),weights_only=False)

f_R = f_R_reparams(z_0=z_0, b_prime_min=b_prime_min, b_max=b_max, alpha=alpha)

conditions = [CustomCondition(f_R.v_reparam),
              CustomCondition(f_R.r_prime_reparam)
              ]

r_prime_net_index = -1
v_net_index = 2

r_prime = BundleSolution1D([nets[r_prime_net_index]], [conditions[-1]])
v = BundleSolution1D([nets[v_net_index]], [conditions[0]])

def H_ML(z, theta, **kwarg):
    b, Om_m_0, H_0 = theta[:-1]

    z_prime = 1 - (z/z_rescale)
    b_prime = b/b_max

    if kwarg.pop('post', False):
        z_prime = z_prime*np.ones_like(b_prime)
        no_reshape = False
    else:
        shape, no_reshape = _shape_manager(z, b, Om_m_0)
        b_prime = b_prime*shape
        Om_m_0 = Om_m_0*shape

    r_prime_sol = r_prime(z_prime, b_prime, Om_m_0, to_numpy=True, no_reshape=no_reshape)

    v_sol = v(z_prime, b_prime, Om_m_0, to_numpy=True, no_reshape=no_reshape)

    out = H_0 * np.sqrt(((1 - Om_m_0)/2)*np.exp(r_prime_sol)/v_sol)

    if no_reshape:
        out = out[0][0]
    return out

condition_INT = [BundleIVP(t_0=0, u_0=0)]
sol_INT = BundleSolution1D([net_INT[0]], [condition_INT[0]])

def INT(z, b, Om_m_0):
    shape, no_reshape = _shape_manager(z, b, Om_m_0)

    z_prime = z/z_d_L_rescale
    b_prime = (b/b_max)*shape
    Om_m_0 = Om_m_0 * shape

    out = sol_INT(z_prime,
                  b_prime,
                  Om_m_0,
                  to_numpy=True,
                  no_reshape=no_reshape
                  )

    if no_reshape:
        out = out[0][0]
    return out


def m_b_tilde(zCMB, zHEL, theta):
    b, Om_m_0, H_0, Mabs = theta

    d_L = c * (1 + zHEL) * INT(zCMB, b, Om_m_0)

    out = 25.0 + 5.0 * np.log10(d_L) + Mabs

    return out



#%%
if __name__ == '__main__':
    z_examples = np.linspace(0, z_0)
    b_example = 2
    Om_m_0_example = 0.3
    H_0_example = 67
    M_example = -19.4

    theta_example = [b_example, Om_m_0_example, H_0_example, M_example]

    # print(m_b_tilde(1, 1, theta_example))
    # print(H_ML(1, theta_example))

    # plt.plot(z_examples, m_b_tilde(z_examples, z_examples, theta_example))
    plt.plot(z_examples, H_ML(z_examples, theta_example))
    # plt.xscale('log')
    plt.show()