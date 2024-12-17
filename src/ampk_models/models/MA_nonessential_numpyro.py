import numpyro
import numpyro.distributions as dist

def MA_nonessential_numpyro_model(data=None, data_std=None, solver=None):
    """Returns a numpyro model for the MAPK activation model.

    Args:
        data (np.ndarray): The data to fit the model to.
        data_std (np.ndarray): The standard deviation of the data.
        solver (wrapper around a dfrx.Solver): The solver to use for the model.
    """

    # fixed parameters
    param = numpyro.deterministic('pname', val)
    
    # PRIORS
    param = numpyro.sample('pname', dist.Gamma(prior))
   

    # std of likelihood
    if data_std is None:
        data_std = numpyro.sample('data_sigma', dist.LogNormal(0, 0.01))

    # run solver
    params = ()
    if solver is not None:
        predict = solver(params)
    else:
        raise Exception("Solver is not defined")

    # likelihood conditioned on the observations
    numpyro.sample('obs', dist.Normal(data, data_std), obs=data)