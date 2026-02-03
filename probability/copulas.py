import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def amh_copula(u, v, theta):
    """Computes CDF of the Ali–Mikhail–Haq copula.

    .. math::

        F(u,v\mid \theta) = \frac {uv}{1-\theta (1-u)(1-v)},
        \quad \theta\in [-1,1]
    """
    return u*v / (1-theta*(1-u)*(1-v))

def amh_copula_density(u, v, theta):
    """Computes PDF of the Ali–Mikhail–Haq copula.

    .. math::

        \frac{\partial^2 F}{\partial u\, \partial v}
        = \frac{\partial^2}{\partial u\, \partial v}
        \frac{u v}{1 - \theta (1 - u) (1 - v)}
        = \frac{\theta^2 (u (-v) + u + v - 1) - \theta (u v + u + v - 2) - 1}
        {(\theta (u - 1) (v - 1) - 1)^3}
    """
    numerator = theta**2 * (-u*v + u + v - 1) - theta * (u*v + u + v - 2) - 1
    denominator = (theta * (u-1) * (v-1) - 1)**3
    return numerator / denominator

def _gumbel_log_term(u, v, theta):
    return (-np.log(u))**theta + (-np.log(v))**theta

def gumbel_copula(u, v, theta):
    """Computes the CDF of the Gumbel copula.
    """
    return np.exp(-_gumbel_log_term(u,v,theta)**(1/theta))

def gumbel_copula_density(u,v,theta):
    """Computes the PDF of the Gumbel copula.
    """
    logs_to_theta = _gumbel_log_term(u,v,theta)
    a = 1/(u*v)
    b = (-np.log(u))**(theta-1) * (-np.log(v))**(theta-1)
    c = gumbel_copula(u, v, theta)
    d = logs_to_theta**(1/theta - 2)
    e = logs_to_theta**(1/theta) + theta - 1
    return a*b*c*d*e

def clayton_copula(u,v,theta):
    """Computes the CDF of the Clayton copula."""
#     u,v = map(np.asarray, (u,v))
    if theta:
        y = u**(-theta) + v**(-theta) - 1
        cdf = np.where(y>0, y**(-1/theta), 0)
        return cdf
    else:
        # independent copula CDF when theta=0
        return u*v

def clayton_copula_density(u,v,theta):
    """Computes the PDF of the Clayton copula."""
#     u,v = map(np.asarray, (u,v))
    if theta:
        y = u**(-theta) + v**(-theta) - 1
        pdf = np.where(y>0, (theta+1)*(u*v)**(-theta-1)*y**(-1/theta-2), 0)
        return pdf
    else:
        # constant density for independent copula when theta=0
        return u*v / (u*v)

def symmetrized(f):
    """Returns the symmetric part of a function f(u, v, theta) on [0,1]x[0,1],
    where u and v are the variables in [0,1], and theta is an extra parameter.
    The function f is symmetrized over both u and v.
    """
    def symmetrized_f(u, v, theta):
        return (f(u, v, theta) + f(1-u, 1-v, theta))/2
    return symmetrized_f

def plot_copula_densities(density_function, theta, num_vs, ax=None):
    """Plots cross-sections of the density f(u, v, theta) of a copula function
    with fixed parameter theta, and the density of the mixture of the cross
    sections.
    """
    if ax is None:
        ax = plt.gca()

    vs = np.linspace(0.001, 0.999, num_vs)
    u = np.linspace(0.001, 0.999, 300)
    mixture_density = sum(density_function(u,v,theta) for v in vs) / num_vs

    integrals = []

    # This works because v is in [0,1]
    ax.set_prop_cycle('color', [plt.cm.seismic(v) for v in vs])

    for v in vs:
        ax.plot(u, density_function(u,v,theta))
        integrals.append(integrate.quad(
            lambda u: density_function(u,v,theta), 0.0001,0.9999))

    ax.plot(u, mixture_density, lw=3);
    ax.set_ylim(0,20)
    return ax, integrals
