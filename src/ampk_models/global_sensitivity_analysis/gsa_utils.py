import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfrx
from jax import lax

# jitable function to run the system to steady-state
@jax.jit
def solve_to_steady_state(params, rhs, y0, event_rtol=1e-12, event_atol=1e-12):
    solver=dfrx.Kvaerno5()
    event = dfrx.SteadyStateEvent(rtol=1e-12, atol=1e-12)
    stepsize_controller = dfrx.PIDController(rtol=1e-10, atol=1e-10)
    t0 = 0.0
    t1 = 5e6
    dt0 = 1e-10 # initial time step

    # initial solve
    sol = dfrx.diffeqsolve(
        rhs, 
        solver, 
        t0, 
        t1, # max time if ss check is not met
        dt0, 
        y0, 
        # saveat=saveat, 
        discrete_terminating_event=event,
        stepsize_controller=stepsize_controller,
        args=params,
	    throw=False)
	    # max_steps=None)
    
    return sol # returns the final state

# jitable function to compute the qois
@jax.jit
def single_model_eval(params, rhs_basal, rhs_stress, y0, ampkar_idx, 
                      pampkar_idx, event_rtol, event_atol):
    # update y0 for AMPKAR
    y0 = y0.at[ampkar_idx].set(params[-1])

    # find basal steady-state
    sol_basal = solve_to_steady_state(params, rhs_basal, y0, event_rtol=event_rtol, event_atol=event_atol)

    # apply stimulus and run again
    sol_stress = solve_to_steady_state(params, rhs_stress, sol_basal.ys[-1,:])

    # use ratio of pAMPKAR/AMPKARtot
    # note that params[-1] is the IC of AMPKAR, which the equal to AMPKARtot
    basal = sol_basal.ys[0,pampkar_idx]/params[-1]
    stress = sol_stress.ys[0,pampkar_idx]/params[-1]
    change = stress - basal
    norm_change = change / basal
    
    return jnp.array([norm_change, change, sol_basal.ts[0], sol_stress.ts[0]])

@jax.jit
def single_model_eval_nansafe(params, rhs_basal, rhs_stress, y0, ampkar_idx, 
                              pampkar_idx, event_rtol=1e-12, event_atol=1e-12):
    pred = jnp.sum(jnp.isnan(params))
    false_fun = lambda params: single_model_eval(params, rhs_basal, rhs_stress, 
                                                 y0, ampkar_idx, pampkar_idx, 
                                                 event_rtol=1e-12, event_atol=1e-12)
    true_fun = lambda params: jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan])
    return lax.cond(pred, true_fun, false_fun, params)

# function to compute all MA params from sampled params
def compute_MA_params(params):
    return (1.0, # kOnAMP
            params[0], # kOffAMP
            1.0 , # kOnADP
            params[1], # kOffADP
            1.0, # kOnATP
            params[2], # kOffATP
            params[3], # kOnCaMKK
            1.0, # kOffCaMKK
            params[4], # kPhosCaMKK
            params[5], # kOnLKB1
            1.0, # kOffLKB1
            params[6], # kPhosLKB1
            params[7], # kOnPP
            1.0, # kOffPP
            params[8], # kDephosPP
            params[9], # kOnAMPK
            1.0, # kOffAMPK
            params[10], # kPhosAMPK
            params[11], # kOnPP1
            1.0, # kOffPP1
            params[12],
            params[13]) # kDephosPP1

def compute_newmech_MA_params(params):
    return (1.0, # kOnAMP
            params[0], # kOffAMP
            params[1], # kOnCaMKK
            1.0, # kOffCaMKK
            params[2], # kPhosCaMKK
            params[3], # kOnLKB1
            1.0, # kOffLKB1
            params[4], # kPhosLKB1
            params[5], # kOnPP
            1.0, # kOffPP
            params[6], # kDephosPP
            params[7], # kOnAMPK
            1.0, # kOffAMPK
            params[8], # kPhosAMPK
            params[9], # kOnPP1
            1.0, # kOffPP1
            params[10], # kDephosPP1
            1.0, # alpha
            params[11], # beta
            params[12], # AMPKAR_0 = AMPKARtot
            )

def mass_action_to_michaelis_menten(k_rev, k_for, k_cat, Et):
	"""function to convert from mass action to michaelis menten parameters
	Returns:
		- Vmax = k_cat * Et
		- Km = (k_rev+k_cat)/k_for
	"""
	return k_cat*Et, (k_rev+k_cat)/k_for

def michaelis_menten_to_mass_action(Vmax, Km, Et, k_rev=1.0, k_cat=None):
	"""function to convert from michaelis menten to mass action parameters
	Requires the user to specify Et and k_rev or k_for
	Returns:
		- k_cat = Vmav / Et
		- k_rev = Km*k_for - k_cat 
			or
		- k_for =  (k_rev+k_cat)/Km
	"""
	if k_cat is None:
		k_cat = Vmax/Et 
	
	if k_rev is None: # return k_rev
		return k_cat, Km*k_for - k_cat
	else: # return k_for
		return k_cat, (k_rev+k_cat)/Km