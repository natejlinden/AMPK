"""
    Nathaniel Linden (UCSD MAE)
    Created: November 26th, 2024
"""
import numpyro
import numpyro.distributions as dist

def MM_numpyro_model(y=None, y_std=None, solver=None):
    """Returns a numpyro model for the MM model.

    Args:
        y (np.ndarray): The data to fit the model to.
        y_std (np.ndarray): The standard deviation of the data.
        solver (wrapper around a dfrx.Solver): The solver to use for the model.
    """

    # fixed parameters
    
    
    # PRIORS
    V_max = numpyro.sample('V_max', dist.Gamma(1.6836, rate=1.7057))
    K_m = numpyro.sample('K_m', dist.Gamma(1.6836, rate=1.7057))
    
    # std of likelihood
    if y_std is None:
        y_std = numpyro.sample('data_sigma', dist.LogNormal(0, 0.01))

    # run solver
    params = (V_max, K_m)
    if solver is not None:
        predict = solver(params)
    else:
        raise Exception("Solver is not defined")

    # likelihood conditioned on the observations
    numpyro.sample('obs', dist.Normal(predict, y_std), obs=y)