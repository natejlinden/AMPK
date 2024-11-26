"""
    Nathaniel Linden (UCSD MAE)
    Created: November 26th, 2024
"""
import numpyro
import numpyro.distributions as dist

def Hill_numpyro_model(data=None, data_std=None, solver=None):
    """Returns a numpyro model for the Hill model.

    Args:
        data (np.ndarray): The data to fit the model to.
        data_std (np.ndarray): The standard deviation of the data.
        solver (wrapper around a dfrx.Solver): The solver to use for the model.
    """

    # fixed parameters
    
    
    # PRIORS
    V_max = numpyro.sample('V_max', dist.Gamma(2.301, rate=209.419))
    K_m = numpyro.sample('K_m', dist.Gamma(3.781, rate=496.016))
    n = numpyro.sample('n', dist.Gamma(2.302, 29.088))
    
    # std of likelihood
    if data_std is None:
        data_std = numpyro.sample('data_sigma', dist.LogNormal(0, 0.01))

    # run solver
    params = (V_max, K_m, n)
    if solver is not None:
        predict = solver(params)
    else:
        raise Exception("Solver is not defined")

    # likelihood conditioned on the observations
    numpyro.sample('obs', dist.Normal(predict, data_std), obs=data)