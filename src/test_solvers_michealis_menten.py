import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sunode
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import time
import sys

sys.path.insert(0, './ampk_models')
import mich_ment_MA as MM


# Nathaniel Linden (UCSD MAE)
# This script compares the runtime for three ODE solvers when solving
# the simple mass-action model for an enzyme catalyzed reaction

# general stuff for all three approaches
Nsolves = 100 # number of simulations to run
savedir = './solve_test/' # where to save outputs
tspan = (0, 15)
tvals = np.linspace(tspan[0], tspan[1], 50)

####################################################
# Scipy integrate.solve_ivp with the 'BDF' method
# default tolerances: atol=1e-6  rtol=1e-3
####################################################
def model_rhs_scipy(t, y, p):
    '''
    p = [kOn, kOff, kCat]
    y = [S, E, ES, P]
    '''
    # FLUXES
    J1 = p[0]*y[0]*y[1] - p[1]*y[2]
    J2 = p[2]*y[2]

    # now return the odes for each state variable
    dydt = np.zeros(y.shape)

    dydt[0] = -J1 #dSdt
    dydt[1] = -J1 + J2 #dEdt
    dydt[2] = J1 - J2 #dESdt
    dydt[3] = J2 #dPdt

    return dydt

# define needed things
y0 = np.array([10.0, 5.0, 0.0, 0.0])
p = np.array([10.0, 2.0, 0.5])
tvals = np.linspace(0, 15, 50)

# solve once to start solution
sol_scipy = solve_ivp(model_rhs_scipy, tspan, y0, method='BDF', args=[p])

# now solve Nsolve times and time each one, store all times
times_scipy = np.zeros((Nsolves,1))
for i in range(Nsolves):
    start = time.time()
    _ = solve_ivp(model_rhs_scipy, tspan, y0, method='BDF', args=[p])
    end = time.time()
    times_scipy[i] = end - start

print('The average time for scipy was: ' + str(np.mean(times_scipy)))

####################################################
# sunode with the 'BDF' method
# default tolerances: 
####################################################
# load the model from the correct file
# load the necessary dictionaries
params = MM.mich_ment_get_params()
states =MM.mich_ment_get_states()

# create sunode problem
problem = sunode.SympyProblem(
    params=params,
    states=states,
    rhs_sympy=MM.mich_ment_RHS_odeEnzyme,
    # specify variables to take gradients wrt
    derivative_params=(),
)

# set up solver
solver = sunode.solver.Solver(problem, solver='BDF')

# Initial conditions
y0 = np.zeros((), dtype=problem.state_dtype)
y0['S'] = 10.0
y0['P'] = 0.0
y0['E'] = 5.0
y0['ES'] = 0.0

# nominal parameter values
solver.set_params_dict({
    'kOn': 10.0,
    'kOff': 2.0,
    'kCat': 0.5,
})

# run solver
yout = solver.make_output_buffers(tvals)
solver.solve(t0=0, tvals=tvals, y0=y0, y_out=yout)

# now solve Nsolve times and time each one, store all times
times_sunode = np.zeros((Nsolves,1))
for i in range(Nsolves):
    solver.set_params_dict({ # want to simulate changing params before each run
        'kOn': 10.0,
        'kOff': 2.0,
        'kCat': 0.5,
    })
    start = time.time()
    solver.solve(t0=0, tvals=tvals, y0=y0, y_out=yout)
    end = time.time()
    times_sunode[i] = end - start

print('The average time for sunode was: ' + str(np.mean(times_sunode)))


####################################################
# diffrax with the Kaverno5() solver and the NewtonNonlinearSolver
# Also use the PIDcontroller adaptive time stepper default tolerances 
####################################################

# first we need to define the model appropriately
class model_rhs_diffrax(eqx.Module):
    kOn: float
    kOff: float
    kCat: float

    def __call__(self, t, y, args):
        # FLUXES
        J1 = self.kOn*y[0]*y[1] - self.kOff*y[2]
        J2 = self.kCat*y[2]

        dSdt = -J1 #dSdt
        dEdt = -J1 + J2 #dEdt
        dESdt = J1 - J2 #dESdt
        dPdt = J2 #dPdt

        return jnp.stack([dSdt, dEdt, dESdt, dPdt])
    
# now we want to define a function to solve the ode as a function of the params
@jax.jit # use the jax.jit decorated to jit compile the function
def main(kOn, kOff, kCat):
    sub_prod = model_rhs_diffrax(kOn, kOff, kCat)
    terms = diffrax.ODETerm(sub_prod)
    t0 = 0.0
    t1 = 15.0
    y0 = jnp.array([10.0, 5.0, 0.0, 0.0])
    dt0 = 0.0002
    solver = diffrax.Kvaerno5(nonlinear_solver=diffrax.NewtonNonlinearSolver())
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, 15, 50))
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    return sol

# run solver to store the solution and compile everything
sol_diffrax = main(p[0], p[1], p[2])

# now solve Nsolve times and time each one, store all times
times_diffrax = np.zeros((Nsolves,1))
for i in range(Nsolves):
    start = time.time()
    main(p[0], p[1], p[2])
    end = time.time()
    times_diffrax[i] = end - start

print('The average time for diffrax was: ' + str(np.mean(times_diffrax)))
