"""
    Nathaniel Linden (UCSD MAE)
    Created: April 24th, 2023

    This file contains the functions for a model of MAPK activation. That model makes the 
    following high-level assumptions:
        - allow double adenine nucleotide AMPK binding
        - use only mass action kinetics
        - the reaction mechanism reflects specific activation and inhibition of
            AMPK and phos/dephos by AXPs

    The get_params functions returns a dictionary of the parameters in the correct
    format.

    The get_states function returns a dictionary od the parameters in the correct
    format.

    The RHS function is written following the syntax specified for the pymc/sunode
    package. See docs here https://sunode.readthedocs.io/en/latest/without_pymc.html
"""
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def MA_single_numpyro_model(data=None, data_std=None, solver=None):
    """Returns a numpyro model for the MAPK activation model.

    Args:
        data (np.ndarray): The data to fit the model to.
        data_std (np.ndarray): The standard deviation of the data.
        solver (wrapper around a dfrx.Solver): The solver to use for the model.
    """

    # fixed parameters
    kOnAMP = numpyro.deterministic('kOnAMP', jnp.array(1.0))
    kOnADP = numpyro.deterministic('kOnADP', jnp.array(1.0))
    kOnATP = numpyro.deterministic('kOnATP', jnp.array(1.0))
    kOnCaMKK = numpyro.deterministic('kOnCaMKK', jnp.array(1.0))
    kOffCaMKK = numpyro.deterministic('kOffCaMKK', jnp.array(1.32e-2))
    kPhosCaMKK = numpyro.deterministic('kPhosCaMKK', jnp.array(8.1e-4))
    kOnLKB1 = numpyro.deterministic('kOnLKB1', jnp.array(1.0))
    kOffLKB1 = numpyro.deterministic('kOffLKB1', jnp.array(1.396))
    kPhosLKB1 = numpyro.deterministic('kPhosLKB1', jnp.array(3.92e-3))
    kOnPP = numpyro.deterministic('kOnPP', jnp.array(1.0))
    kOffPP = numpyro.deterministic('kOffPP', jnp.array(5.6e-1))
    kDephosPP = numpyro.deterministic('kDephosPP', jnp.array(1.1e-4))
    kOnAMPK = numpyro.deterministic('kOnAMPK', jnp.array(1.0))
    kOnPP1 = numpyro.deterministic('kOnPP1', jnp.array(1.0))
    
    # PRIORS
    # kOffAMP = numpyro.sample('kOffAMP', dist.Gamma(2.301, rate=209.419))
    # kOffADP = numpyro.sample('kOffADP', dist.Gamma(3.781, rate=496.016))
    # kOffATP = numpyro.sample('kOffATP', dist.Gamma(2.302, 29.088))
    # kOffAMPK = numpyro.sample('kOffAMPK', dist.Gamma(2.302, 6.167))
    # kPhosAMPK = numpyro.sample('kPhosAMPK', dist.Gamma(3.781, 38751.36))
    # kOffPP1 = numpyro.sample('kOffPP1', dist.Gamma(2.302, 0.935))
    # kDephosPP1 = numpyro.sample('kDephosPP1', dist.Gamma(3.781, 6763.874))
    kOffAMP = numpyro.sample('kOffAMP', dist.LogNormal(loc=-2.386, scale=7.802e-01))
    kOffADP = numpyro.sample('kOffADP', dist.LogNormal(loc=-2.897, scale=7.802e-01))
    kOffATP = numpyro.sample('kOffATP', dist.LogNormal(loc=-4.121e-01, scale=7.802e-01))
    kOffAMPK = numpyro.sample('kOffAMPK', dist.LogNormal(loc=1.139, scale=7.802e-01))
    kPhosAMPK = numpyro.sample('kPhosAMPK', dist.LogNormal(loc=-7.255, scale=7.802e-01))
    kOffPP1 = numpyro.sample('kOffPP1', dist.LogNormal(loc=3.025, scale=7.802e-01))
    kDephosPP1 = numpyro.sample('kDephosPP1', dist.LogNormal(loc=-5.510, scale=7.802e-01))

    # std of likelihood
    if data_std is None:
        data_std = numpyro.sample('data_sigma', dist.LogNormal(0, 0.01))

    # run solver
    params = (kOnAMP, kOffAMP, kOnADP, kOffADP, kOnATP, kOffATP, kOnCaMKK, kOffCaMKK, kPhosCaMKK, kOnLKB1, kOffLKB1, kPhosLKB1, kOnPP, kOffPP, kDephosPP, kOnAMPK, kOffAMPK, kPhosAMPK, kOnPP1, kOffPP1, kDephosPP1)
    if solver is not None:
        predict = solver(params)
    else:
        raise Exception("Solver is not defined")

    # likelihood conditioned on the observations
    # with numpyro.plate('data', len(data) if data is not None else len(data_std)):
    if data is not None:
        numpyro.sample('obs', dist.Normal(predict, data_std), obs=data)
    else:
        numpyro.sample('obs', dist.Normal(predict, data_std))