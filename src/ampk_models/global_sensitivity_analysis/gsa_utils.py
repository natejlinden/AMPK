import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfrx
from jax import lax

# def set_init_conds(problem_ode, state_names):
#     """ Function to set non-zero initial conditions for the models."""
#     y0 = np.zeros((), dtype=problem_ode.state_dtype)
#     to_set = ['AMP', 'ADP', 'ATP', 'AMPK', 'CaMKK', 'LKB1', 'PP', 'PP1', 'AMPKAR']
#     not_set = list(set(state_names) - set(to_set))
    
#     #update nonzero ICs
#     y0['AMP'] = 2e-5   # 'AMP' mM
#     y0['ADP'] = 1.3e-1 # 'ADP mM
#     y0['ATP'] = 8.2   # 'ATP mM
#     # # free AMPK
#     y0['AMPK'] = 0.6   # 'AMPK mM
#     y0['CaMKK'] = 1.0   # 'CaMKK_AMPK mM
#     y0['LKB1'] = 1.0   # 'CaMKK_AMPK mM
#     y0['PP'] = 1.0   # 'CaMKK_AMPK mM
#     y0['AMPKAR'] = 0.1   # 'AMPKAR mM
#     y0['PP1'] = 1.0   # 'CaMKK_AMPK mM

#     # update zero ICs
#     for state in not_set:
#         y0[state] = 0.0
        
#     return y0

# def single_model_eval(params, problem_ode, solver, y0, tvals, state_names, full_output=True):
#     """ Function to evaluate the model at a single set of parameters. 
#     Runs system to steady state, then applies 2-DG stimulus and runs to steady state again.

#     Evaluates the chosen QoIs at the steady state after the perturbation.

#     Inputs:
#         params: list of parameters to evaluate the model at
#         problem_ode: ODE model for sunode
#         solver: solver for sunode
#         y0: initial conditions for the model
#         tvals: time values to evaluate the model
#         state_names: list of state names for the model
#     """

#     sol_init, times_init = sim_to_steady_state(params, 1300.0, problem_ode, 
#                         solver, y0, tvals, state_names, make_p_dict_fun_MA, full_output, thresh=1e-6,)
#     y0_new = np.zeros((), dtype=problem_ode.state_dtype)
#     for state in state_names:
#         y0_new[state] = sol_init[state][-1]

#     sol_stim, times_stim = sim_to_steady_state(params, 4.5, problem_ode, 
#                         solver, y0_new, tvals, state_names, make_p_dict_fun_MA, full_output, thresh=1e-6,)

#     # now compute the output QoIs
#     # pAMPKAR/AMPKAR steady-state before perturb
#     pAMPKAR_AMPKAR_ss = sol_init.view(problem_ode.state_dtype)['pAMPKAR'][-1] / sol_init.view(problem_ode.state_dtype)['AMPKAR'][-1]

#     # change after glycolysis perturbation
#     pAMPKAR_AMPKAR_final_ss = sol_stim.view(problem_ode.state_dtype)['pAMPKAR'][-1] / sol_stim.view(problem_ode.state_dtype)['AMPKAR'][-1]
#     pAMPKAR_AMPKAR_delta = (pAMPKAR_AMPKAR_final_ss - pAMPKAR_AMPKAR_ss)/pAMPKAR_AMPKAR_ss

#     if full_output:
#         return np.array([pAMPKAR_AMPKAR_ss, pAMPKAR_AMPKAR_delta]), sol_init, sol_stim, times_init, times_stim
#     else:
#         return np.array([pAMPKAR_AMPKAR_ss, pAMPKAR_AMPKAR_delta])

# def sim_to_steady_state(params, glyco_flux, problem_ode, 
#                         solver, y0, tvals, state_names,
#                         make_p_dict_fun, full_output=True,
#                         thresh=1e-6, t_int_add=100, t_cnt_add=200, max_add_iter=1e6):
#     """ Function to simulate the model to steady state.
    
#     Checks for steady state after simulation for time defined by tvals. Steady state 
#     is evaluated by the check_steady_state function which checks in the mean gradient of the last
#     4 time points normalized to the mean of the last for time points is smaller than the threshold. 
#     If steady state is not reached, will run 100s simulations until steady state is reached.

#     Inputs:
#     ------- 
#             params: list of parameters
#             glyco_flux: glycolysis flux
#             problem_ode: ODE problem for sunode
#             solver: solver for sunode
#             y0: initial conditions
#             tvals: time points to simulate
#             state_names: list of state names
#             make_p_dict_fun: function to make parameter dictionary
#             thresh: threshold for steady state checking
#             t_int_add (optional): time interval to add to tvals if steady state
#                                     is not reached (default: 100s)
#             t_cnt_add (optional): number of time points to add to tvals if steady state
#                                     is not reached (default: 200)
#             max_add_iter (optional): maximum number of iterations to add to tvals if not steady state
#                                     (default: 50)
    
#     Returns:
#     -------
#             solution: solution to the ODE evaluated at tvals plus any additional time points 
#                         if steady state is not reached
#             times: time points for solution (tvals + any additional time points)
#     """
#     ######################
#     # IC and parameters
#     # update relevant initial conditions 
#     y0['AMPKAR'] = params[13]

#     # set the parameters
#     solver.set_params_dict(make_p_dict_fun(params, glyco_flux))

#     # list to store all sols
#     sol_list = []

#     ######################
#     # simulation
#     # initial simulation for tvals
#     yout = solver.make_output_buffers(tvals)
#     solver.solve(t0=0, tvals=tvals, y0=y0, y_out=yout)
#     if full_output: # only store if full output 
#         sol_list.append(yout)

#     # check for steady state
#     pAMPKAR_AMPKAR = yout.view(problem_ode.state_dtype)['pAMPKAR'] / yout.view(problem_ode.state_dtype)['AMPKAR']
#     ss_check = check_steady_state(pAMPKAR_AMPKAR, thresh)

#     # if not steady state, run in 100 second intervals until steady state
#     n_additional_sims = 0 # count number of additional simulations
#     while not ss_check and n_additional_sims <= max_add_iter:
#         n_additional_sims += 1
#         # update initial conditions with previous simulation
#         y0_new = np.zeros((), dtype=problem_ode.state_dtype)
#         for state in state_names:
#             y0_new[state] = yout.view(problem_ode.state_dtype)[state][-1]

#         # run again
#         tvals_inter = np.linspace(0, t_int_add, t_cnt_add)
#         yout = solver.make_output_buffers(tvals_inter)
#         solver.solve(t0=0, tvals=tvals_inter, y0=y0_new, y_out=yout)
#         if full_output: # only store if full output
#             sol_list.append(yout)

#         # check for steady state
#         pAMPKAR_AMPKAR = yout.view(problem_ode.state_dtype)['pAMPKAR'] / yout.view(problem_ode.state_dtype)['AMPKAR']
#         ss_check = check_steady_state(pAMPKAR_AMPKAR, thresh)

#         # print params if n_additional sims>5e4
#         if n_additional_sims==5e4:
#             print(params)
#         # if we reach steady state, and we are not storing the full output, store the final sim
#         if ss_check and not full_output:
#             sol_list.append(yout)
    
#     if n_additional_sims == max_add_iter:
#         print('WARNING: max number of additional simulations reached. Steady state not reached.')

#     # compile sol_list into one solution trajecotry
#     # first need the overall shape
#     shape = (sum([sol.shape[0] for sol in sol_list]),1)
#     solution = np.zeros(shape, dtype=problem_ode.state_dtype)
#     for state in state_names: # iterate over sols
#         temp = [sol.view(problem_ode.state_dtype)[state] for sol in sol_list]
#         solution[state] = np.vstack(temp)
#     # store the simulation times
#     # WARNING: times is meaningless if full_output is False so set to None
#     if full_output:
#         times = np.hstack([tvals, 
#                        np.linspace(tvals[-1], tvals[-1] + n_additional_sims*t_int_add, 
#                                    n_additional_sims*t_cnt_add)])
#     else:
#         times = None
    
#     return solution, times

# def check_steady_state(traj, thresh, n=4):
#     """ Function to check if a trajectory is at steady state.
    
#     Checks if the mean gradient of the last n time points normalized to the
#         mean of the last n times is less than the threshold.

#     Inputs:
#     -------
#             traj: trajectory to check
#             thresh: threshold for steady state checking
#             n (optional): number of time points to check (default: 4)
#     """
#     n *= -1
#     grad = np.abs(np.gradient(traj, axis=0))
#     return np.mean(grad[n:])/np.mean(traj[n:]) <= thresh


# def make_p_dict_fun_MA(params, glyco_flux):     
#     """ Returns the parameters dictionary for the MA model.
    
#     Inputs: params - vector of variable parameters
#             glyco_flux - glycolysis flux
#     Outputs: p_dict - dictionary of parameters
#     """
#     return {
#         'kOnAMP': 1.0, # 1/(mM s) 
#         'kOffAMP': params[0], 
#         'kOnADP': 1.0 ,
#         'kOffADP': params[1],
#         'kOnATP': 1.0, 
#         'kOffATP': params[2],
#         'kOnCaMKK': params[3],
#         'kOffCaMKK': 1.0, 
#         'kPhosCaMKK': params[4], 
#         'kOnLKB1': params[5], 
#         'kOffLKB1': 1.0,
#         'kPhosLKB1': params[6], 
#         'kOnPP': params[7], 
#         'kOffPP': 1.0,
#         'kDephosPP': params[8],
#         'kOnAMPK': params[9],
#         'kOffAMPK': 1.0,
#         'kPhosAMPK': params[10],
#         'kOnPP1': params[11],
#         'kOffPP1': 1.0,
#         'kDephosPP1': params[12],
#         'kGly': glyco_flux,
#         'kHydro':1.4e-3,
#         'kForAK':40.44,
#         'kRevAK':1.1e-3,
#         'VmaxOxPhos':0.5,
#         'Kadp': 5.8e-2,
#         'n': 2.568,
#     }

# def set_init_conds(problem_ode, state_names):
#     """ Function to set non-zero initial conditions for the models."""
#     y0 = np.zeros((), dtype=problem_ode.state_dtype)
#     to_set = ['AMP', 'ADP', 'ATP', 'AMPK', 'CaMKK', 'LKB1', 'PP', 'PP1', 'AMPKAR']
#     not_set = list(set(state_names) - set(to_set))
    
#     #update nonzero ICs
#     y0['AMP'] = 2e-5   # 'AMP' mM
#     y0['ADP'] = 1.3e-1 # 'ADP mM
#     y0['ATP'] = 8.2   # 'ATP mM
#     # # free AMPK
#     y0['AMPK'] = 0.6   # 'AMPK mM
#     y0['CaMKK'] = 1.0   # 'CaMKK_AMPK mM
#     y0['LKB1'] = 1.0   # 'CaMKK_AMPK mM
#     y0['PP'] = 1.0   # 'CaMKK_AMPK mM
#     y0['AMPKAR'] = 0.1   # 'AMPKAR mM
#     y0['PP1'] = 1.0   # 'CaMKK_AMPK mM

#     # update zero ICs
#     for state in not_set:
#         y0[state] = 0.0
        
#     return y0

# def set_init_conds_MM(problem_ode, state_names):
#     # Initial conditions
#     y0 = np.zeros((), dtype=problem_ode.state_dtype)
#     y0['AMP'] = 2e-5   # 'AMP' mM
#     y0['ADP'] = 1.3e-1 # 'ADP mM
#     y0['ATP'] = 8.2   # 'ATP mM
#     y0['AMPK'] = 0.6   # 'AMPK mM
#     y0['AMPKAR'] = 1e-3   # 'AMPKAR mM

#     return y0


# jitable function to run the system to steady-state
@jax.jit
def solve_to_steady_state(params, rhs, y0, event_rtol=1e-12, event_atol=1e-12):
    solver=dfrx.Kvaerno5()
    event = dfrx.SteadyStateEvent(rtol=1e-12, atol=1e-12)
    stepsize_controller = dfrx.PIDController(rtol=1e-10, atol=1e-10)
    t0 = 0.0
    t1 = 5e6 # 3000.0 # 1000 seconds
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

    # # compute the ratio of pAMPKAR/AMPKAR at the end of the stress simulation
    # basal_ratio = sol_basal.ys[0,pampkar_idx]/sol_basal.ys[0,ampkar_idx]
    # stress_ratio = sol_stress.ys[0,pampkar_idx]/sol_stress.ys[0,ampkar_idx]
    # norm_change = (stress_ratio - basal_ratio) / basal_ratio

    # use ratio of pAMPKAR/AMPKARtot
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