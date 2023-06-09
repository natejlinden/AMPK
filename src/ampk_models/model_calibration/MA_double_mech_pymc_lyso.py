import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfrx
import json
import pandas as pd
# from diffrax import LinearInterpolation, ODETerm, Kvaerno5, PIDController, SaveAt, diffeqsolve

import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

import pymc as pm
import pymc.sampling.jax as pmsj
import arviz as az
import matplotlib.pyplot as plt

# import models
import sys
sys.path.insert(0, '../odes')
sys.path.insert(0, '../global_sensitivity_analysis')
import ampk_MA_double_mech_diffrax as model
from gsa_utils import *

import tensorflow_probability.substrates.jax as tfp
jax.scipy.special.erfcx = tfp.math.erfcx

# use 64 bit precision
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

RANDOM_SEED = 342
rng = np.random.default_rng(RANDOM_SEED)
###########################################
# get user inputs
############################################
data_file = '../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz'
dir = './'
base_name = 'MA_double_mech'
model_info_json = '../global_sensitivity_analysis/MA_double.json'
nominals_file = '../global_sensitivity_analysis/nominal_params_MA.csv'

############################################
# DATA
############################################
# load data 
data = np.load(data_file)
times = data['times']
zero_idx = int(np.where(times==0.0)[0]) # we only want values after the 2-DG stimulus
times = times[zero_idx:]
mean_data = data['mean'][zero_idx:]
std_data = data['std_constant']

############################################
# SETUP DIFFRAX solver
############################################
# states and initial conditions
with open(model_info_json, 'r') as file:
        model_data = json.load(file)

# unpack loaded model data dictionary
state_names = model_data['state_names']
ampkar_states = model_data['ampkar_states']
pampkar_states = model_data['pampkar_states']
n_states = len(state_names)
y0_states_to_set = model_data['y0']['set_states']
y0_vals_to_set = model_data['y0']['set_ics']

# get the indices of the states
ampkar_idxs = [state_names.index(item) for item in ampkar_states]
pampkar_idxs = [state_names.index(item) for item in pampkar_states]
ampkar_idx = state_names.index('AMPKAR')
pampkar_idx = state_names.index('pAMPKAR')

################################################
# Nominal parameters
################################################
nominals_file = pd.read_csv(nominals_file)
nominals = {}
for key, val in zip(nominals_file['parameter'].to_list(),  nominals_file['value'].to_list()):
     nominals[key] = val

# fix AMPKAR_0 because it is not identifiable and it will be tricky to set in the model
AMPKAR_0 = nominals['AMPKAR_0']

# fixed parameters
kOffCaMKK =  nominals['kOffCaMKK']
kPhosCaMKK = nominals['kPhosCaMKK']
kOffLKB1 =   nominals['kOffLKB1']
kPhosLKB1 =  nominals['kPhosLKB1']
kOffPP =     nominals['kOffPP']
kDephosPP =  nominals['kDephosPP']
kOffPP1 =   nominals['kOffPP1']

# Set initial conditions
y0 = np.zeros(n_states)
for state, val in zip(y0_states_to_set, y0_vals_to_set):
    y0[state_names.index(state)] = val
y0[ampkar_idx] = AMPKAR_0
# random seed for reproducibility
seed = np.random.seed(seed=2048)

################################################
# Elicited prior parameters
################################################
# states and initial conditions
prior_params_json = './MA_prior_params.json'
with open(prior_params_json, 'r') as file:
        prior_params = json.load(file)

################################################
#                   Model RHS                  #
################################################
# metabolism_params
metab_parms_basal = {'kGly': 0.5,'kHydro':0.1,
                     'VforAK': 14.66, 'KeqAK': 2.21, 'kmm': 0.32, 'kmd': 0.35, 'kmt': 0.27,
                     'VmaxOxPhos':0.5,'Kadp': 5.8e-2,'n': 2.568,}
metab_parms_stress = {'kGly': 0.005,'kHydro':0.1,
                      'VforAK': 14.66, 'KeqAK': 2.21, 'kmm': 0.32, 'kmd': 0.35, 'kmt': 0.27,
                     'VmaxOxPhos':0.5,'Kadp': 5.8e-2,'n': 2.568,}

rhs = model.vector_field(**metab_parms_basal)
rhs_stress = model.vector_field(**metab_parms_stress)
rhs = dfrx.ODETerm(rhs)
rhs_stress = dfrx.ODETerm(rhs_stress)

################################################
# Jax functions for the ODE solution and the gradient
################################################
saveat=dfrx.SaveAt(ts=times)
t0 = 0.0
t1 = times[-1]
solver=dfrx.Kvaerno5()
event_rtol = 1e-8
event_atol = 1e-8
event = dfrx.SteadyStateEvent(event_rtol, event_atol)
stepsize_controller = dfrx.PIDController(rtol=1e-8, atol=1e-8)

def sol_op_jax(KdAMP, KdADP, KdATP, kOffAMPK, kPhosAMPK, kDephosPP1):
    args = compute_MA_params([KdAMP, KdADP, KdATP, kOffCaMKK, kPhosCaMKK, kOffLKB1, kPhosLKB1, kOffPP,
               kDephosPP, kOffAMPK, kPhosAMPK, kOffPP1, kDephosPP1, AMPKAR_0]) # tuple to pass to the solver!
    # first run the system to the estimates basal steady state
    # this gives us the initial condition for the stress response
    sol_basal = dfrx.diffeqsolve(
            rhs, solver, t0=t0, t1=1e5, dt0=1e-8,
            y0=y0, # known initial conditions!, we can change later
            stepsize_controller=stepsize_controller,
            args=args, throw=False,
        )
    
    # now we purturb the system by changing the glycolysis paramter
    # and solve the model at the experimental time points
    sol_stress = dfrx.diffeqsolve(
            rhs_stress, solver, t0=t0, t1=t1, dt0=1e-8,
            y0=sol_basal.ys[-1,:], # known initial conditions!, we can change later
            saveat=saveat, stepsize_controller=stepsize_controller,
            args=args, # max_steps=6000,
            throw=False,
        )
    
    # now compute the QOI which should match the experimental data
    AMPKAR_tot_basal = np.sum(sol_basal.ys[:,ampkar_idxs], axis=1)
    AMPKAR_tot_stress = np.sum(sol_stress.ys[:,ampkar_idxs], axis=1)
    pAMPKAR_basal = np.sum(sol_basal.ys[:,pampkar_idxs], axis=1)
    pAMPKAR_stress = np.sum(sol_stress.ys[:,pampkar_idxs], axis=1)
    qoi = (pAMPKAR_stress/AMPKAR_tot_stress)-(pAMPKAR_basal[-1]/AMPKAR_tot_basal[-1])
    
    return qoi

# get a jitted (compiled) version of the function
jitted_sol_op_jax = jax.jit(sol_op_jax)

# vector jacobian product (vjp)
def vjp_sol_op_jax(KdAMP, KdADP, KdATP, kOffAMPK, kPhosAMPK, kDephosPP1, output_grads):
    _, vjp_fn = jax.vjp(sol_op_jax, KdAMP, KdADP, KdATP, kOffAMPK, kPhosAMPK, kDephosPP1)
    return vjp_fn(output_grads)

# get a jitted (compiled) version of the function
jitted_vjp_sol_op_jax = jax.jit(vjp_sol_op_jax)

# # Uncomment below to check the jitted functions
KdAMP = 2.5e-3
KdADP = 1.5e-3
KdATP = 1.8e-2
kOffAMPK = 8.49e-2
kPhosAMPK = 1.92e-5
kDephosPP1 = 1.1e-2
sol = jitted_sol_op_jax(KdAMP, KdADP, KdATP, kOffAMPK, kPhosAMPK, kDephosPP1)
grad = jitted_vjp_sol_op_jax(KdAMP, KdADP, KdATP, kOffAMPK, kPhosAMPK, kDephosPP1, sol)

print('The ODE solution is:')
print(sol)
print('The gradients evaluated at the solution are:')
print(grad)

################################################
# PyTensor Ops
################################################
class SolOp(Op):
    def make_node(self, KdAMP, KdADP, KdATP, kOffAMPK, kPhosAMPK, kDephosPP1):
        # Convert our inputs to symbolic variables
        inputs = [pt.as_tensor_variable(KdAMP), pt.as_tensor_variable(KdADP), pt.as_tensor_variable(KdATP),
                  pt.as_tensor_variable(kOffAMPK), pt.as_tensor_variable(kPhosAMPK),
                  pt.as_tensor_variable(kDephosPP1)]
        # assume output is always a float64
        outputs = [pt.dvector()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        KdAMP, KdADP, KdATP, kOffAMPK, kPhosAMPK, kDephosPP1 = inputs
        result = jitted_sol_op_jax(KdAMP, KdADP, KdATP, kOffAMPK, kPhosAMPK, kDephosPP1)
        outputs[0][0] = np.asarray(result, dtype="float64")
    
    def grad(self, inputs, output_gradients):
        KdAMP, KdADP, KdATP, kOffAMPK, kPhosAMPK, kDephosPP1 = inputs
        (gz,) = output_gradients
        return vjp_sol_op(KdAMP, KdADP, KdATP, kOffAMPK, kPhosAMPK, kDephosPP1, gz)

class VJPSolOp(Op):
        def make_node(self, KdAMP, KdADP, KdATP, kOffAMPK, kPhosAMPK, kDephosPP1, output_grads):
            inputs = [pt.as_tensor_variable(KdAMP), pt.as_tensor_variable(KdADP), pt.as_tensor_variable(KdATP),
                        pt.as_tensor_variable(kOffAMPK), pt.as_tensor_variable(kPhosAMPK),
                        pt.as_tensor_variable(kDephosPP1), pt.as_tensor_variable(output_grads)]
            outputs = [inputs[0].type(), inputs[1].type(), inputs[2].type(), inputs[3].type(),
                       inputs[4].type(), inputs[5].type()] # grad wrt to each input parameter
            return Apply(self, inputs, outputs)
        
        def perform(self, node, inputs, outputs):
            KdAMP, KdADP, KdATP, kOffAMPK, kPhosAMPK, kDephosPP1, output_grads = inputs
            result = jitted_vjp_sol_op_jax(KdAMP, KdADP, KdATP, kOffAMPK, kPhosAMPK, kDephosPP1, output_grads)
            outputs[0][0] = np.asarray(result[0], dtype="float64") 
            outputs[1][0] = np.asarray(result[1], dtype="float64") 
            outputs[2][0] = np.asarray(result[2], dtype="float64") 
            outputs[3][0] = np.asarray(result[3], dtype="float64") 
            outputs[4][0] = np.asarray(result[4], dtype="float64") 
            outputs[5][0] = np.asarray(result[5], dtype="float64")

sol_op = SolOp()
vjp_sol_op = VJPSolOp()

try:
    pytensor.gradient.verify_grad(sol_op, (KdAMP, KdADP, KdATP, kOffAMPK, kPhosAMPK, kDephosPP1,), 
                                  rng=np.random.default_rng(), eps=1e-10, n_tests=4)
except pytensor.gradient.GradientError as err:
    print('Did not pass unit test! Investigate more! \nThe stack trace was: \n')
    print(Exception, err)

@jax_funcify.register(SolOp)
def sol_op_jax_funcify(op, **kwargs):
    return sol_op

@jax_funcify.register(VJPSolOp)
def vjp_sol_op_jax_funcify(op, **kwargs):
    return vjp_sol_op

################################################
# PyMC model
################################################
lyso_model = pm.Model()

with lyso_model:
    # priors
    KdAMP = pm.LogNormal("KdAMP", mu=prior_params["KdAMP"]["mu"], tau=prior_params["KdAMP"]["tau"])
    KdADP = pm.LogNormal("KdADP", mu=prior_params["KdADP"]["mu"], tau=prior_params["KdADP"]["tau"])
    KdATP = pm.LogNormal("KdATP", mu=prior_params["KdATP"]["mu"], tau=prior_params["KdATP"]["tau"])
    kOffAMPK = pm.LogNormal("kOffAMPK", mu=prior_params["kOffAMPK"]["mu"], tau=prior_params["kOffAMPK"]["tau"])
    kPhosAMPK = pm.LogNormal("kPhosAMPK", mu=prior_params["kPhosAMPK"]["mu"], tau=prior_params["kPhosAMPK"]["tau"])
    kDephosPP1 = pm.LogNormal("kDephosPP1", mu=prior_params["kDephosPP1"]["mu"], tau=prior_params["kDephosPP1"]["tau"])

    
    # evaluate the model at the parameters
    # computes the ampkar signal after 2-DG stimulus
    ampkar_signal = sol_op(KdAMP, KdADP, KdATP, kOffAMPK, kPhosAMPK, kDephosPP1)

    # loglikelihood
    # note that sigma comes from the data!
    llike = pm.Normal("llike", mu=ampkar_signal, sigma=std_data, observed=mean_data)

################################################
# PyMC sampling with the numpyro (jax-based) NUTS sampler
################################################
# prior predictive sampling
# with lyso_model:
#    prior_checks = pm.sample_prior_predictive(samples=200, random_seed=rng)
# az.to_netcdf(prior_checks, dir + base_name + '/lyso_prior_predictive.nc')

# # posterior samples
# with lyso_model:
#     # draw 4000 posterior samples
#     # numpyro NUTS
#     idata = pm.sample(draws=4000, chains=4, idata_kwargs={'log_likelihood':True})
#     # idata = pmsj.sample_numpyro_nuts(draws=4000, chains=4, idata_kwargs={'log_likelihood':True})
# az.to_netcdf(idata, dir + base_name + '/lyso_posterior.nc')

# # posterior predictive samples
# with lyso_model:
#     # draw 4000 posterior samples
#     # numpyro NUTS
#     posterior_checks = pm.sample_posterior_predictive(idata, idata_kwargs={'log_likelihood':True}, random_seed=rng)
# az.to_netcdf(idata, dir + base_name + '/lyso_posterior_predictive.nc')


ip = lyso_model.initial_point()
print(ip)
logp_fn = lyso_model.compile_fn(lyso_model.logp(sum=False))
print('Non-jax logp', logp_fn(ip))

logp_fn = lyso_model.compile_fn(lyso_model.logp(sum=False), mode="JAX")
print('Jax logp', logp_fn(ip))

dlogp_fn = lyso_model.compile_fn(lyso_model.dlogp())
print('non-jax dlog_p', dlogp_fn(ip))

dlogp_fn = lyso_model.compile_fn(lyso_model.dlogp(), mode="JAX")
print('jax dlog_p', dlogp_fn(ip))


