"""
    Nathaniel Linden (UCSD MAE)
    Created: November 26th, 2024
"""
import numpyro
import numpyro.distributions as dist

def MA_numpyro_model(y=None, y_std=None, solver=None):
    """Returns a numpyro model for the MA model.

    Args:
        y (np.ndarray): The data to fit the model to.
        y_std (np.ndarray): The standard deviation of the data.
        solver (wrapper around a dfrx.Solver): The solver to use for the model.
    """

    # fixed parameters
    # k_f = numpyro.deterministic('k_f', 1.0)
    
    
    # PRIORS
    k_f = numpyro.sample('k_f', dist.Gamma(2.0, rate=0.5))
    # k_r = numpyro.sample('k_r', dist.Gamma(1.6836, rate=1.7057))
    k_r = numpyro.sample('k_r', dist.Gamma(2.0, rate=0.5))
    k_cat = numpyro.sample('k_cat', dist.Gamma(1.6836, rate=1.7057))
    
    # std of likelihood
    if y_std is None:
        y_std = numpyro.sample('y_sigma', dist.LogNormal(0, 0.01))

    # run solver
    params = (k_f, k_r, k_cat)
    if solver is not None:
        predict = solver(params)
    else:
        raise Exception("Solver is not defined")

    # likelihood conditioned on the observations
    with numpyro.plate('data', len(y) if y is not None else 10):
        numpyro.sample('obs', dist.Normal(predict, y_std), obs=y)