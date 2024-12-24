from neurodiffeq.conditions import BaseCondition
import torch
import numpy as np


class CustomCondition(BaseCondition):
    r"""A custom condition where the parametrization is custom-made by the user to comply with
    the conditions of the differential system.

    :param parametrization:
        The custom parametrization that complies with the conditions of the differential system. The first input
        is the output of the neural network. The rest of the inputs of the parametrization are
        the inputs to the neural network with the same order as in the solver.
    :type parametrization: callable
    """

    def __init__(self, parametrization):
        super().__init__()
        self.parameterize = parametrization

    def enforce(self, net, *coords):
        r"""Enforces this condition on a network with `N` inputs

        :param net: The network whose output is to be re-parameterized.
        :type net: `torch.nn.Module`
        :param coords: Inputs of the neural network.
        :type coords: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`
        .. note::
            This method overrides the default method of ``neurodiffeq.conditions.BaseCondition`` .
            In general, you should avoid overriding ``enforce`` when implementing custom boundary conditions.
        """
        def net_fun(*coords):
            r"""The neural network as a function

            :param coords: Inputs of the neural network.
            :type coords: `torch.Tensor`.
            :return: The output or outputs of the network.
            :rtype: list[`torch.Tensor`, `torch.Tensor`,...] or `torch.Tensor`.
            """
            outs = net(torch.cat([*coords], dim=1))
            out_units = net.NN[-1].out_features
            if out_units > 1:
                outs = [outs[:, index].view(-1, 1) for index in range(out_units)]
            return outs
        return self.parameterize(net_fun, *coords)


class f_R_reparams:
    def __init__(self, z_0, b_prime_min, b_max, alpha, pert_type='Default'):
        self.z_0 = z_0
        self.b_prime_min = b_prime_min
        self.b_max = b_max
        self.alpha = alpha
        self.pert_type = pert_type

    def pert(self, z_prime, b_prime, Om_m_0):

        if self.pert_type == 'Default':
            p = (1 - torch.exp(-z_prime)) * (1 - torch.exp(-self.alpha*(b_prime - self.b_prime_min)))

        elif self.pert_type == 'exponent_square':
            p = (1 - torch.exp(-z_prime)) * (1 - torch.exp(-self.alpha*((b_prime - self.b_prime_min) ** 2)))

        elif self.pert_type == 'square':
            p = (1 - torch.exp(-z_prime)) * ((1 - torch.exp(-self.alpha*(b_prime - self.b_prime_min))) ** 2)
        else:
            raise ValueError(f'{self.pert_type} is not a valid entry for pert_type')

        return p

    def x_reparam(self, net_fun, z_prime, b_prime, Om_m_0):

        x_N = net_fun(z_prime, b_prime, Om_m_0)

        out = self.pert(z_prime, b_prime, Om_m_0) * x_N
        return out

    def y_reparam(self, net_fun, z_prime, b_prime, Om_m_0):

        z = self.z_0 * (1 - z_prime)

        y_N = net_fun(z_prime, b_prime, Om_m_0)

        y_hat = (Om_m_0*((1 + z)**3) + 2*(1 - Om_m_0))/(2*(Om_m_0*((1 + z)**3) + 1 - Om_m_0))

        out = y_hat + self.pert(z_prime, b_prime, Om_m_0) * y_N
        return out

    def v_reparam(self, net_fun, z_prime, b_prime, Om_m_0):

        z = self.z_0 * (1 - z_prime)

        v_N = net_fun(z_prime, b_prime, Om_m_0)

        v_hat = (Om_m_0*((1 + z)**3) + 4*(1 - Om_m_0))/(2*(Om_m_0*((1 + z)**3) + 1 - Om_m_0))

        out = v_hat + self.pert(z_prime, b_prime, Om_m_0) * v_N
        return out

    def Om_reparam(self, net_fun, z_prime, b_prime, Om_m_0):

        z = self.z_0 * (1 - z_prime)

        Om_N = net_fun(z_prime, b_prime, Om_m_0)

        Om_hat = Om_m_0*((1 + z)**3)/(Om_m_0*((1 + z)**3) + 1 - Om_m_0)

        out = Om_hat + self.pert(z_prime, b_prime, Om_m_0) * Om_N
        return out

    def r_prime_reparam(self, net_fun, z_prime, b_prime, Om_m_0):

        z = self.z_0 * (1 - z_prime)

        r_prime_N = net_fun(z_prime, b_prime, Om_m_0)

        r_hat = (Om_m_0*((1 + z)**3) + 4*(1 - Om_m_0))/(1 - Om_m_0)

        if isinstance(r_hat, torch.Tensor):
            r_prime_hat = torch.log(r_hat)
        else:
            r_prime_hat = np.log(r_hat)

        out = r_prime_hat + self.pert(z_prime, b_prime, Om_m_0) * r_prime_N
        return out


def _shape_manager(z, b, Om_m_0):
    no_reshape = False
    if isinstance(z, (float, int)):
        shape = 1
        if isinstance(b, (float, int)) and isinstance(Om_m_0, (float, int)):
            no_reshape = True
    else:
        shape = np.ones_like(z)
    return shape, no_reshape