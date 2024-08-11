import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfrx
import lineax as lnx
import pymc as pm
from pymc.sampling.jax import sample_numpyro_nuts
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify
import json
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import sys

sys.path.insert(0, './')
from pymc_jax_ode import *

rng = np.random.default_rng(seed=1234)

def two_comp(t, state, args):
    """ two compartment model"""
    k1e, k12, k21, b = args # unpack params
    x, y = state # unpack state
    d_x = -(k1e + k12) * x + k21 * y + b
    d_y = k12 * x - k21 * y

    return (d_x, d_y)

ode_dfrx = dfrx.ODETerm(two_comp)

@jax.jit
def solve_traj(params, y0, times):
    """ simulates a model over the specified time interval and returns the calculated values. """

    dt0=1e-3 # initial time step
    # solver = dfrx.Kvaerno5() # solver
    solver = dfrx.Kvaerno5(root_finder=dfrx.VeryChord(rtol=1e-9, atol=1e-9, linear_solver=lnx.AutoLinearSolver(well_posed=False))) # solver
    stepsize_controller=dfrx.PIDController(rtol=1e-9, atol=1e-9, pcoeff=0.2, icoeff=0.4, dcoeff=0)

    # times = jnp.arange(t0, t1, t_step) # time points to save
    saveat=dfrx.SaveAt(ts=times)

    sol = dfrx.diffeqsolve(
        ode_dfrx, 
        solver, 
        times[0], 
        times[-1],
        dt0, 
        tuple(y0), 
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        max_steps=None,
        args=params,
        adjoint=dfrx.RecursiveCheckpointAdjoint(checkpoints=100),
        throw=False,)
    
    return jnp.array(sol.ys)

params = (2/3, 4/3, 1, 1)
y0 = (0.5, 0.5)

# run the model
times = jnp.arange(0, 15, 0.2)
reference_sol = solve_traj(params, y0, times)

# add noise to the data
noise = rng.normal(0, 0.1, reference_sol.shape)
noisy_data = reference_sol + noise

# plot the results
fig, ax = plt.subplots(2,1, figsize = (3, 2), sharex=True)
ax[0].plot(times, reference_sol[0, :], label='x1', color='C0')
ax[0].scatter(times, noisy_data[0, :], 4, color='C0', alpha=0.5)
ax[0].set_ylabel('x1')
ax[0].set_title('Two-compartment model')
ax[0].set_ylim(0, 2.5)

ax[1].plot(times, reference_sol[1, :], label='x2', color='C1')
ax[1].scatter(times, noisy_data[1, :], 4, color='C1', alpha=0.5)
ax[1].set_ylabel('x2')
ax[1].set_ylim(0, 2.5)

times_no_ic = times[1:]

data_std = 0.1 * np.ones_like(noisy_data)

def sol_op_jax(*params):
    """jax function to solve the model using diffrax"""
    return solve_traj(params, y0, times)

sol_op_jax_jitted = jax.jit(sol_op_jax)

def vjp_sol_op_jax(gz, *params):
    _, vjp_fn = jax.vjp(sol_op_jax, *params)
    return vjp_fn(gz)

vjp_sol_op_jax_jitted = jax.jit(vjp_sol_op_jax)

vjp_sol_op = VJPSolOp(vjp_sol_op_jax_jitted)
sol_op = SolOp(sol_op_jax_jitted, vjp_sol_op)

# register with jax
@jax_funcify.register(SolOp)
def sol_op_jax_funcify(op, **kwargs):
    return sol_op_jax

@jax_funcify.register(VJPSolOp)
def vjp_sol_op_jax_funcify(op, **kwargs):
    return vjp_sol_op_jax

# build the pymc model
model = pm.Model()
with model:
    # priors
    alpha = pm.Lognormal('k1e', mu=2/3, sigma=0.2)
    beta = pm.Lognormal('k12', mu=1.0, sigma=2.5)
    gamma = pm.Lognormal('k21', mu=1.0, sigma=2.5)
    # delta = pm.Gamma('b',  alpha=2.0, beta=0.5)

    data_sigma = pm.HalfNormal('data_sigma', sigma=1)
 
    # predict
    prediction = sol_op(alpha, beta, gamma, 1.0)

    # likelihood
    llike = pm.Normal("llike", mu=prediction[:, 1:], sigma=data_sigma, observed=noisy_data[:, 1:])

#with model:
#   prior_checks = pm.sample_prior_predictive(samples=1000, random_seed=rng)


#print("Testing PyMC NUTS sampler")
#with model:
#    trace = pm.sample(chains=1, cores=4, tune=100, draws=100,
#    random_seed=rng, target_accept=0.95)

#print("Testing VI sampler")
#with model:
#    trace_vi = pm.fit(method='advi', n=6000, obj_optimizer=pm.adagrad(learning_rate=0.1), progressbar=True)

print("Testing Numpyro NUTS sampler")
with model:
    trace_numpyro = sample_numpyro_nuts(chains=4, tune=100, draws=100,
    random_seed=rng, target_accept=0.95, chain_method="vectorized")

print("all sampers passed")
