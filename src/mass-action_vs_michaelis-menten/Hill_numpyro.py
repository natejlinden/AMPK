"""
    Nathaniel Linden (UCSD MAE)
    Created: November 26th, 2024
"""
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def Hill_numpyro_model(y=None, y_std=None, solver=None):
    """Returns a numpyro model for the Hill model.

    Args:
        data (np.ndarray): The data to fit the model to.
        y_std (np.ndarray): The standard deviation of the data.
        solver (wrapper around a dfrx.Solver): The solver to use for the model.
    """

    # fixed parameters
    
    
    # PRIORS
    V_max = numpyro.sample('V_max', dist.Gamma(1.69, rate=1.71)) # 95% prior mass [1e-3, 2.0]
    K_m = numpyro.sample('K_m', dist.Gamma(5.97, rate=2.16)) # 95% prior mass [0.8, 5.0]
    # n = numpyro.sample('n', dist.Gamma(19.68, rate=14.26)) # 95% prior mass [0.8, 2.0]
    n = numpyro.sample('n', dist.TruncatedNormal(loc=1.75, scale=4.167e-01, low=1.0)) # 95% prior mass [1.0, 2.5]

    # std of likelihood
    if y_std is None:
        y_std = numpyro.sample('data_sigma', dist.LogNormal(0, 0.01))

    # run solver
    params = (V_max, K_m, n)
    if solver is not None:
        predict = solver(params)
    else:
        predict = jnp.zeros_like(y_std)
        # raise Exception("Solver is not defined")

    # likelihood conditioned on the observations
    numpyro.sample('obs', dist.Normal(predict, y_std), obs=y)