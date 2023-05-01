import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import cmdstanpy as stan 
import math
from cmdstanpy import CmdStanModel


stan_file = 'comfortModel.stan'
model = CmdStanModel(stan_file=stan_file)
print(model)
print("_____________________________________1__________________________________________")
print('modle information is ', model.exe_info())
print("______________________________________1.1_________________________________________")

data_file = 'data_file.json'
print('data = ', data_file)
print("_____________________________________1.2__________________________________________")
fit = model.sample(data=data_file, show_console= True)
# access model variable by name
print("______________________________________2_________________________________________")
print(fit.stan_variable('therm_comfort'))
# print(fit.stan_variable('comfort'))
print("______________________________________3_________________________________________")
print(fit.draws_pd('therm_comfort')[:3])
print(fit.draws_xr('therm_comfort'))

print("______________________________________4_________________________________________")
# access all model variables
for k, v in fit.stan_variables().items():
    print(f'{k}\t{v.shape}')
print("______________________________________5_________________________________________")
# access the sampler method variables
for k, v in fit.method_variables().items():
    print(f'{k}\t{v.shape}')
print("______________________________________6_________________________________________")
# access all Stan CSV file columns
print(f'numpy.ndarray of draws: {fit.draws().shape}')

print("______________________________________7_________________________________________")
fit.draws_pd()
print("______________________________________8_________________________________________")
print(fit.metric_type)
print("______________________________________9_________________________________________")
print(fit.metric)
print("______________________________________10_________________________________________")
print(fit.step_size)
print()
print("______________________________________11_________________________________________")
print()
print(fit.metadata.cmdstan_config['model'])
print("______________________________________12_________________________________________")
print()
print(fit.metadata.cmdstan_config['seed'])

print("______________________________________13_________________________________________")
print()
print(fit.metadata.stan_vars_cols.keys())

print("______________________________________14_________________________________________")
print()
print(fit.metadata.method_vars_cols.keys())


fit.summary()
print("______________________________________15_________________________________________")
print()

print(fit.diagnose())
print("______________________________________15_________________________________________")
MLH = model.optimize(data=data_file)
print(fit.stan_variable('therm_comfort'))
print("______________________________________END_________________________________________")
print(" Ending ________________ Every thing is OK")