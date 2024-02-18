#!/usr/bin/env python3

# NOTE: This script requires line search settings to run

from numpy import array, dot
import numpy as np
from surrogate_classes import LineSearch, ParameterStructure, ParameterHessian
from surrogate_macros import load_E_err, surrogate_diagnostics, \
     generate_surrogate, plot_surrogate_pes, plot_surrogate_bias, get_var_eff, nexus_qmcpack_analyzer
from parameters import forward, backward, forward_subspace, backward_subspace, get_pes_job, get_pes_springs_job, dmc_pes_job
from matplotlib import pyplot as plt
from surrogate_classes import LineSearchIteration
import os
from copy import deepcopy
from nexus import obj
from functools import partial

calc_type = 'saddle'
# calc_type = 'eqm1'
# calc_type = 'eqm2'
run_dmc = True
add_sigma = False
window_frac = 0.5 # use 0.5 for saddle, 0.25 for eqm
epsilon_p = [0.01, 0.05, 0.05, 0.05]
shift_params = [0.05, 0.25, 0.05, 0.05] # not used for DMC
n_ls_iter = 4
n_warmup = 2
poly_fit_order = 'pf4'

os.system('rm -r __pycache__ log_no_noise log_noise log_dmc convergence_no_noise convergence_noise convergence_dmc pes_plots')
os.system('mkdir pes_plots')

if run_dmc:
    convergence_dirname = f'convergence_dmc_{calc_type}'
    ls_dirname = 'ls_dmc'
    log_fname = f'log_dmc_{calc_type}'
else:
    if add_sigma:
        convergence_dirname = f'convergence_noise_{calc_type}'
        ls_dirname = 'ls_noise'
        log_fname = f'log_noise_{calc_type}'
    else:
        convergence_dirname = f'convergence_no_noise_{calc_type}'
        ls_dirname = 'ls_no_noise'
        log_fname = f'log_no_noise_{calc_type}'
cmd = f'mkdir {convergence_dirname}'
os.system(cmd)

f_log = open(log_fname, 'w')

write_str = f"""SETTINGS
------------------------------
epsilon_p = {epsilon_p}\n
n_ls_iter = {n_ls_iter}\n
n_warmup = {n_warmup}\n
window_frac = {window_frac}\n
shift_params = {shift_params}\n
run_dmc = {run_dmc}\n
add_sigma = {add_sigma}\n
poly_fit_order = {poly_fit_order}\n
------------------------------\n\n
"""
f_log.write(write_str)

pos_fname = f'{calc_type}_dft.xyz'

pos_array = np.loadtxt(pos_fname, skiprows=2, usecols=(1, 2, 3))
structure = ParameterStructure(
    forward = forward,
    backward = backward,
    params = forward(pos_array),
    elem = ['C'] + 3*['H'] + 2*['F'],
    units = 'A',
    tol = 1e-5
)

write_str = "starting structure: " + str(structure.params) + "\n"
f_log.write(write_str)
f_log.write("\n")

hessian_path_name = f'{calc_type}/hessian'
hessian = ParameterHessian(structure = structure)
hessian.compute_fdiff(
    mode      = 'nexus',
    path      = hessian_path_name,
    dp        = [0.01] * len(structure.params), # without scaling, they are unit vectors
    pes_func  = get_pes_job,
    load_func = load_E_err
    )

write_str = "Hessian eigenvalues: " + str(hessian.Lambda) + "\n"
f_log.write(write_str)
write_str = "Hessian directions (eigenvectors^T): \n" + str(hessian.get_directions()) + "\n\n"
f_log.write(write_str)

surrogate_path_name = f'{calc_type}/surrogate'
surrogate = generate_surrogate(
    path = surrogate_path_name,
    mode = 'nexus',
    # pes_func = get_pes_springs_job,
    pes_func = get_pes_job,
    load_func = load_E_err,
    structure = hessian.structure,
    hessian = hessian,
    fit_kind = poly_fit_order,
    window_frac = window_frac, #GRI window_frac means maximum displacement in units of lambda of each direction
    M = 17,
    fname = 'surrogate.p',
    )

surrogate.run_jobs(interactive=False)
surrogate.load_results(set_target=True)

for i in range(len(hessian.Lambda)):
    surrogate.ls(i).target_x0 = surrogate.ls(i).compute_bias_of(W = surrogate.ls(i).Lambda * 0.01)[0][0]

surrogate.optimize(
    N = 500,
    kind = 'ls',
    M = 7,
    fit_kind = poly_fit_order,
    epsilon_p = epsilon_p,
    reoptimize = False,
)

surrogate.write_to_disk('surrogate.p', overwrite = False)
surrogate_diagnostics(surrogate)
plot_surrogate_pes(surrogate)
surrogate_pes_fname = f'pes_plots/{calc_type}.pdf'
plt.savefig(surrogate_pes_fname)

ls_path_name = f'{calc_type}/{ls_dirname}'

if not run_dmc:
    lsi = LineSearchIteration(
        path = ls_path_name,
        mode = 'nexus',
        surrogate = surrogate,
        # pes_func = get_pes_springs_job,
        pes_func = get_pes_job,
        load_func = load_E_err,
        # targets = surrogate.structure.params,
        shift_params = shift_params,
        )

    for i in range(n_ls_iter):
        lsi.pls(i).run_jobs(interactive=False)
        lsi.pls(i).load_results(add_sigma=add_sigma)
        lsi.propagate(i)

    # Add the final eqm structure
    lsi.pls(n_ls_iter).run_jobs(interactive=False, eqm_only=True)
    lsi.pls(n_ls_iter).load_eqm_results(add_sigma=add_sigma)
#GRI END SECTION

#GRI USE THIS SECTION FOR DMC SEARCH
else:
    var_eff = get_var_eff(
        surrogate.structure,
        dmc_pes_job,
        path = f'{calc_type}/dmc/test',
        suffix = '/dmc/dmc.in.xml',
    )
    
    lsi = LineSearchIteration(
        path = ls_path_name,
        mode = 'nexus',
        surrogate = surrogate,
        pes_func = dmc_pes_job,
        pes_args = {'var_eff': var_eff}, # supply var_eff to target right sigma
        load_func = nexus_qmcpack_analyzer,
        load_args = {'suffix': '/dmc/dmc.in.xml' } # point to right file
    )
    
    for i in range(n_ls_iter):
        lsi.pls(i).run_jobs(interactive=False)
        lsi.pls(i).load_results()
        lsi.propagate(i)
    
    # Add the final eqm structure
    lsi.pls(n_ls_iter).run_jobs(interactive=False, eqm_only=True)
    lsi.pls(n_ls_iter).load_eqm_results()
#GRI END SECTION

count = 0.
mean_sum = np.zeros_like(np.array(structure.params))
squared_err_sum = np.zeros_like(np.array(structure.params))
ener_sum = 0.
ener_err_squared_sum = 0.

for i_p, p in enumerate(lsi.get_params()[0]):
    p_original_param_space = p
    
    # get error for parameter p
    err_p = lsi.get_params()[1][i_p]
    p_upper = p + err_p
    p_lower = p - err_p
    
    write_str = "ls iter " + str(i_p) + ":\n    " + str(p_original_param_space) + "\n"
    f_log.write(write_str)
    write_str = "    +/- " + str(err_p) + ")\n"
    f_log.write(write_str)
    write_str = "    energy: " + str(lsi.pls_list[i_p].structure.value) + " +/- " + str(lsi.pls_list[i_p].structure.error) + " eV\n"
    f_log.write(write_str)
    
    if i_p >= n_warmup:
        count += 1.
        mean_sum += np.array(p_original_param_space)
        squared_err_sum += np.square(np.array(err_p))
        ener_sum += lsi.pls_list[i_p].structure.value
        ener_err_squared_sum += (lsi.pls_list[i_p].structure.error) ** 2.

mean_avg = mean_sum / count
err_avg = np.sqrt(squared_err_sum / count)
ener_avg = ener_sum / count
ener_err_avg = (ener_err_squared_sum / count) ** 0.5

f_log.write("\n")
        
write_str = "mean over last " + str(n_ls_iter - n_warmup + 1) + " iterations:\n    " + str(mean_avg) + "\n"
f_log.write(write_str)
write_str = "    +/- " + str(err_avg) + ")\n"
f_log.write(write_str)
write_str = "    energy: " + str(ener_avg) + " +/- " + str(ener_err_avg) + " eV\n"
f_log.write(write_str)

f_log.close()
