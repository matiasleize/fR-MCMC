import numpy as np 
from matplotlib import pyplot as plt
from ML import H_ML, INT, m_b_tilde

# Set the range of the independent variable:
z_0 = 10.0
z_examples = np.linspace(0, z_0, 50)
#b_example = 2
#Om_m_0_example = 0.3
#H_0_example = 67
#M_example = -19.4

samples = np.random.normal(0,0.1,size = 10)

b_example = 2 * np.ones(10) + samples
Om_m_0_example = 0.3 * np.ones(10) + samples
H_0_example = 67 * np.ones(10) + samples
M_example = -19.4 * np.ones(10) + samples


theta_example = [b_example, Om_m_0_example, H_0_example, M_example]

print(H_ML(0, theta_example, post=True))

# print(m_b_tilde(1, 1, theta_example))
# print(H_ML(1, theta_example))

# plt.plot(z_examples, m_b_tilde(z_examples, z_examples, theta_example))
#plt.plot(z_examples, H_ML(z_examples, theta_example,),'.')
# plt.xscale('log')
plt.show()