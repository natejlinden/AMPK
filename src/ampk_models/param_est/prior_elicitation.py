import preliz as pz
import numpy as np
import json

import matplotlib.pyplot as plt

""""Prior elicitation for the models

Assumes that we will use LogNormal priors for all parameters. 
The bounds for the parameters are taken from the model_info.json file.

Notes on the parameterization from Preliz to PyMC

Preliz parametrizes the lognormal distribution using mu and sigma, where:
-  mean = exp(mu + sigma^2/2)
-  var = (exp(simga^2) - 1) * exp(2*mu + sigma^2)
(https://preliz.readthedocs.io/en/latest/examples/gallery/log_normal.html)

PyMC parametrizes the lognormal distribution using mu and tau, where:
-  mean = exp(mu + 1/tau)
-  var = (exp(1/tau) - 1) * exp(2*mu + 1/tau)
(https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.LogNormal.html)

Thus we need to convert the Preliz mu and sigma to PyMC mu and tau.
mu_PyMC = mu_Preliz
tau_PyMC = sigma_Preliz^(-2)
"""

models = [
    'MA_single', 'MA_double'
]

prob_mass = 0.9

for model in models:
    print(f'Prior elicitation for {model}')

    model_info_file = f'../models/{model}.json'
    with open(model_info_file, 'r') as file:
           model_info = json.load(file)

    if 'prior_params' not in model_info.keys():
        model_info['prior_params'] = {}

    bounds = model_info['param_bounds']
    for param in model_info['params']:
            if len(bounds[param]) > 0:
                result = pz.maxent(pz.LogNormal(), lower=bounds[param][0], 
                                   upper=bounds[param][1], mass=prob_mass, plot=False)
                mu = result.params[0]
                sigma_preliz = result.params[1]
                # convert sigma in Preliz to tau in PyMC (see note above)
                tau_pymc = sigma_preliz**(-2)

                # store in a model_info dictionary
                model_info['prior_params'][param] = {'mu': mu, 'tau': tau_pymc}
            
            if len(bounds[param]) == 0:
                model_info['prior_params'][param] = {}

    # save the updated model_info
    with open(model_info_file, 'w') as file:
        json.dump(model_info, file, indent=4)
        file.truncate()