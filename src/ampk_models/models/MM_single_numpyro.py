import numpyro
import numpyro.distributions as dist

def MM_single_numpyro_model(data=None, data_std=None, solver=None):
    """Returns a numpyro model for the MAPK activation model.

    Args:
        data (np.ndarray): The data to fit the model to.
        data_std (np.ndarray): The standard deviation of the data.
        solver (wrapper around a dfrx.Solver): The solver to use for the model.
    """

    # fixed parameters
    kOnAMP = numpyro.deterministic("kOnAMP", 1.0)
    kOnADP = numpyro.deterministic("kOnADP", 1.0)
    kOnATP = numpyro.deterministic("kOnATP", 1.0)
    KmCaMKK = numpyro.deterministic("KmCaMKK", 1.5e-2)
    KmLKB1 = numpyro.deterministic("KmLKB1", 1.4)
    kPP = numpyro.deterministic("kPP", 1.1e-4)
    KmPP = numpyro.deterministic("KmPP", 6.7e-1)
    kAMPK = numpyro.deterministic("kAMPK", 1.92e-5)
    KmAMPK = numpyro.deterministic("KmAMPK", 8.49e-2)
    kPP1 = numpyro.deterministic("kPP1", 1.1e-4)
    KmPP1 = numpyro.deterministic("KmPP1", 6.7e-1)
    CaMKKtot = numpyro.deterministic("CaMKKtot", 1.0)
    LKB1tot = numpyro.deterministic("LKB1tot", 1.0)
    PPtot = numpyro.deterministic("PPtot", 0.1)
    PP1tot = numpyro.deterministic("PP1tot", 0.1)
    
    # PRIORS
    kOffAMP = numpyro.sample('kOffAMP', dist.LogNormal(loc=-2.386, scale=7.802e-01))
    kOffADP = numpyro.sample('kOffADP', dist.LogNormal(loc=-2.897, scale=7.802e-01))
    kOffATP = numpyro.sample('kOffATP', dist.LogNormal(loc=-4.121e-01, scale=7.802e-01))
    kCaMKK = numpyro.sample('kPhosCaMKK', dist.LogNormal(loc=-3.513, scale=7.802e-01))
    kLKB1 = numpyro.sample('kPhosLKB1', dist.LogNormal(loc=-1.936, scale=7.802e-01))
   

    # std of likelihood
    if data_std is None:
        data_std = numpyro.sample('data_sigma', dist.LogNormal(0, 0.01))

    # run solver
    params = (kOnAMP, kOffAMP, kOnADP, kOffADP, kOnATP, kOffATP, KmCaMKK, kCaMKK, KmLKB1, kLKB1, kPP, KmPP, kAMPK, KmAMPK, kPP1, KmPP1, CaMKKtot, LKB1tot, PPtot, PP1tot)
    if solver is not None:
        predict = solver(params)
    else:
        raise Exception("Solver is not defined")

    # likelihood conditioned on the observations
    with numpyro.plate('data', len(data) if data is not None else len(data_std)):
        numpyro.sample('obs', dist.Normal(predict, data_std), obs=data)