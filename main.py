# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import cmdstanpy as stan
import os
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt 

stan_file = 'F:\pythonCodes\BayesianBN\comfort.stan'
data_file = 'F:\pythonCodes\BayesianBN\data.json'

model = CmdStanModel(stan_file=stan_file)
print( )
print("-------------------------------------1--------------------------------------------------------")
print(model)

print( )
print("-------------------------------------2--------------------------------------------------------")
print(model.exe_info())

print( )
print("-------------------------------------3--------------------------------------------------------")
fit = model.sample(data= data_file, show_console=True)


print( )
print("-------------------------------------4--------------------------------------------------------")
print(fit.stan_variable('mu'))

print( )
print("-------------------------------------5--------------------------------------------------------")
print(fit.draws_pd('mu')[:3])

print( )
print("-------------------------------------6--------------------------------------------------------")
print(fit.draws_xr('mu'))

print( )
print("-------------------------------------7--------------------------------------------------------")
for k, v in fit.stan_variables().items():
    print(f'{k}\t{v.shape}')

print( )
print("-------------------------------------8--------------------------------------------------------")
for k, v in fit.method_variables().items():
    print(f'{k}\t{v.shape}')

print( )
print("-------------------------------------9--------------------------------------------------------")
print(f'numpy.ndarray of draws: {fit.draws().shape}')

print( )
print("-------------------------------------10--------------------------------------------------------")
fit.draws_pd()

print( )
print("-------------------------------------11--------------------------------------------------------")
print(fit.metric_type)

print( )
print("-------------------------------------12--------------------------------------------------------")
print(fit.metric)


print( )
print("-------------------------------------13--------------------------------------------------------")
print(fit.step_size)


print( )
print("-------------------------------------14--------------------------------------------------------")
print(fit.metadata.cmdstan_config['model'])

print( )
print("-------------------------------------15--------------------------------------------------------")
print(fit.metadata.cmdstan_config['seed'])

print( )
print("-------------------------------------16--------------------------------------------------------")
print(fit.metadata.stan_vars_cols.keys())

print( )
print("-------------------------------------17--------------------------------------------------------")
print(fit.metadata.method_vars_cols.keys())

print( )
print("-------------------------------------18--------------------------------------------------------")
fit.summary()

print( )
print("-------------------------------------19--------------------------------------------------------")
print(fit.diagnose())

print( )
print("---------------------------              END         -----------------------------------------")












# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
    
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
