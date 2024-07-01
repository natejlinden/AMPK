import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import equinox as eqx
import diffrax as dfrx
import sys

# add the odes directory to the path
sys.path.insert(0, '../odes')
import ampk_MA_double_mech_diffrax as model

# tell Jax to use 64 bit precision
jax.config.update("jax_enable_x64", True)

# metabolism_params
metab_parms_basal = {
    'kGly': 1300.0,
    'kHydro':1.4e-3,
    'kForAK':40.44,
    'kRevAK':1.1e-3,
    'VmaxOxPhos':0.5,
    'Kadp': 5.8e-2,
    'n': 2.568,}

# rhs 
rhs = model.ampk_MA_double_mech(**metab_parms_basal)
rhs = dfrx.ODETerm(rhs)
# model parameters
nominal_vals_MA = [
    2.5e-3, # KdAMP
    1.5e-3, # KdADP
    1.7e-3, # KdATP
    90.467, # kOnCaMKK = (koff+kphos/Km)
    0.357, # kPhosCaMKK
    0.742, # kOnLKB1
    3.92e-2, # kPhosLKB1
    16.56, # kOnPP
    1.1e-1, # kDephosPP
    1569.59, # kOnAMPK
    6.33, # kPhosAMPK
    16.56, # kOnPP1
    1.1e-1, # kDephosPP1
    1e-3, # AMPKAR
]

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
            params[12]) # kDephosPP1

# initial conditions
state_names = ['AMP', 'ADP', 'ATP', 'AMPK', 'pAMPK', 'AMP_AMPK', 'ADP_AMPK', 'ATP_AMPK', 'AMP_pAMPK', 'ADP_pAMPK', 'ATP_pAMPK', 'AMP_AMP_AMPK', 'AMP_ADP_AMPK', 'AMP_ATP_AMPK', 'ADP_ADP_AMPK', 'ADP_ATP_AMPK', 'ATP_ATP_AMPK', 'AMP_AMP_pAMPK', 'AMP_ADP_pAMPK', 'AMP_ATP_pAMPK', 'ADP_ADP_pAMPK', 'ADP_ATP_pAMPK', 'ATP_ATP_pAMPK', 'CaMKK', 'CaMKK_AMPK', 'CaMKK_AMP_AMPK', 'CaMKK_ADP_AMPK', 'CaMKK_ATP_AMPK', 'CaMKK_AMP_AMP_AMPK', 'CaMKK_AMP_ADP_AMPK', 'CaMKK_AMP_ATP_AMPK', 'CaMKK_ADP_ADP_AMPK', 'CaMKK_ADP_ATP_AMPK', 'CaMKK_ATP_ATP_AMPK', 'LKB1', 'LKB1_AMP_AMPK', 'LKB1_ADP_AMPK', 'LKB1_AMP_AMP_AMPK', 'LKB1_AMP_ADP_AMPK', 'LKB1_ADP_ADP_AMPK', 'PP', 'PP_pAMPK', 'PP_ATP_pAMPK', 'PP_AMP_ATP_pAMPK', 'PP_ADP_ATP_pAMPK', 'PP_ATP_ATP_pAMPK', 'AMPKAR', 'pAMPKAR', 'AMPKAR_AMP_pAMPK', 'AMPKAR_AMP_AMP_pAMPK', 'AMPKAR_AMP_ADP_pAMPK', 'PP1', 'PP1_pAMPKAR']

to_set = ['AMP', 'ADP', 'ATP', 'AMPK', 'CaMKK', 'LKB1', 'PP', 'AMPKAR', 'PP1']
idxs = [state_names.index(item) for item in to_set]

print(idxs)

y0 = np.zeros((53,))
y0[idxs[0]] = 2e-5   # 'AMP' mM
y0[idxs[1]] = 1.3e-1 # 'ADP mM
y0[idxs[2]] = 8.2   # 'ATP mM
y0[idxs[3]] = 0.6   # 'AMPK mM
y0[idxs[4]] = 1.0   # 'CaMKK_AMPK mM
y0[idxs[5]] = 1.0   # 'CaMKK_AMPK mM
y0[idxs[6]] = 1.0   # 'CaMKK_AMPK mM
y0[idxs[7]] = 0.1   # 'AMPKAR mM
y0[idxs[8]] = 1.0   # 'CaMKK_AMPK mM

y0 = jnp.array(y0)
## Solver overhead
solver=dfrx.Kvaerno5()
stepsize_controller = dfrx.PIDController(rtol=1e-8, atol=1e-8)
t0 = 0.0
t1 = 1000.0 # 1000 seconds
times = np.arange(t0, t1, 0.5)
dt0 = 1e-3 # initial time step
saveat=dfrx.SaveAt(ts=times)

# solve
sol_healthy_dfrx = dfrx.diffeqsolve(
    rhs, 
    solver, 
    t0, t1, dt0, 
    y0, 
    saveat=saveat, stepsize_controller=stepsize_controller,
    args=compute_MA_params(nominal_vals_MA))

print(sol_healthy_dfrx)


## Now test a solve to steady-state function
@jax.jit
def solve_to_steady_state(params):
    solver=dfrx.Kvaerno5()
    stepsize_controller = dfrx.PIDController(rtol=1e-8, atol=1e-8)
    t0 = 0.0
    t1 = 1000.0 # 1000 seconds
    times = np.arange(t0, t1, 0.5)
    dt0 = 1e-3 # initial time step
    saveat=dfrx.SaveAt(ts=times)

    # initial solve
    sol = dfrx.diffeqsolve(
        rhs, 
        solver, 
        t0, t1, dt0, 
        y0, 
        saveat=saveat, stepsize_controller=stepsize_controller,
        args=params)