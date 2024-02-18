#!/usr/bin/env python3

# NOTE: This script requires line search settings to run

from numpy import array, dot
import numpy as np
from surrogate_classes import LineSearch, ParameterStructure, ParameterHessian
from surrogate_macros import load_E_err, surrogate_diagnostics, \
     generate_surrogate, plot_surrogate_pes, plot_surrogate_bias, get_var_eff, nexus_qmcpack_analyzer
from parameters import forward, backward, forward_subspace, backward_subspace, get_pes_job, get_pes_springs_job, dmc_pes_job
import orthonormal_subspace
from matplotlib import pyplot as plt
from surrogate_classes import LineSearchIteration
import os
from copy import deepcopy
from nexus import obj
from functools import partial

use_springs = True
spring_const = 0.0
n_intermediate = 5
n_iter = 1 # number of force-free pathway determination iterations
epsilon_p = [0.01, 0.05, 0.05, 0.05]
# epsilon_p = None
n_ls_iter = 4 # number of line search iterations for each image with each pathway determination iteration
n_warmup = 2 # number of initial line search iterations to skip when averaging parameters, make sure >= 1 if |init_params_shift_factor| > 0
structures_dir = 'neb_seed_structures_fmax0.01'
window_frac = 0.5
poly_fit_order = 'pf4'
dmc_enhancement_factor = 16

# offset by a general value in the subspace
# init_params_shift_factor = 0.05 # use this for DFT
init_params_shift_factor = 0.0 # use this for DMC

# alternatively, offset by specific values in parameter space
init_params_shift = None # if specifying, set shifts in parameter space, e.g. ([0.05, 0.25])

run_dmc = True
add_sigma = False

os.system('rm -r __pycache__ log_no_noise log_noise log_dmc convergence_no_noise convergence_noise convergence_dmc pes_plots')
os.system('mkdir pes_plots')

if run_dmc:
    convergence_dirname = 'convergence_dmc'
    ls_dirname = 'ls_dmc'
    log_fname = 'log_dmc'
else:
    if add_sigma:
        convergence_dirname = 'convergence_noise'
        ls_dirname = 'ls_noise'
        log_fname = 'log_noise'
    else:
        convergence_dirname = 'convergence_no_noise'
        ls_dirname = 'ls_no_noise'
        log_fname = 'log_no_noise'
cmd = f'mkdir {convergence_dirname}'
os.system(cmd)

f_log = open(log_fname, 'w')

write_str = f"""SETTINGS
------------------------------
use_springs = {use_springs}\n
spring_const = {spring_const}\n
n_intermediate = {n_intermediate}\n
n_iter = {n_iter}\n
epsilon_p = {epsilon_p}\n
n_ls_iter = {n_ls_iter}\n
n_warmup = {n_warmup}\n
structures_dir = {structures_dir}\n
window_frac = {window_frac}\n
poly_fit_order = {poly_fit_order}\n
init_params_shift_factor = {init_params_shift_factor}\n
init_params_shift = {init_params_shift}\n
run_dmc = {run_dmc}\n
add_sigma = {add_sigma}\n
dmc_enhancement_factor = {dmc_enhancement_factor}
------------------------------\n\n
"""
f_log.write(write_str)

n_total = n_intermediate + 2 # total number of images in the transition pathway,
                             #+2 for initial and final

assert n_intermediate == (len([fname for fname in os.listdir(structures_dir) \
                              if os.path.isfile(os.path.join(structures_dir, fname))]) - 2), \
                              "Error: Number of intermediate structures doesn't match interpolated seed"

def get_tangent(prev_img, next_img, curr_img=None, curr_idx=None):
    # currently only implementing tangent at r_i as (r_{i+1} - r_{i-1})
    # return (next - prev)
    if curr_idx < ((n_total+1)//2 - 1):
        return (next_img - curr_img) / np.linalg.norm(next_img - curr_img)
    elif curr_idx > ((n_total+1)//2 - 1):
        return (curr_img - prev_img) / np.linalg.norm(curr_img - prev_img)
    else:
        d1 = (next_img - curr_img) / np.linalg.norm(next_img - curr_img)
        d2 = (curr_img - prev_img) / np.linalg.norm(curr_img - prev_img)
        return ((d1 + d2) / np.linalg.norm(d1 + d2))

def convert_to_string(value):
    if isinstance(value, np.ndarray):
        return "np.array(" + repr(value.tolist()) + ")"
    return str(value)

if use_springs:
    f_log.write("CAUTION!\n")
    f_log.write("You have set use_springs to True.\n")
    f_log.write("Please make sure you have set the correct\n")
    f_log.write("directory in pes_springs_template.py\n")

structures = obj()
structures_pos_arrays = []

for idx in range(n_total): # +2 for initial and final
    pos_fname = f'{structures_dir}/structure_' + str(idx) + '.xyz'
    pos_array = np.loadtxt(pos_fname, skiprows=2, usecols=(1, 2, 3))
    structures[idx] = ParameterStructure(
        forward = forward,
        backward = backward,
        params = forward(pos_array),
        elem = ['C'] + 3*['H'] + 2*['F'],# + ['Cl'],
        units = 'A',
        tol = 1e-5
    )
    # structures.append(curr_str)
    structures_pos_arrays.append(pos_array)

start_state = backward(structures[0].params)
end_state = backward(structures[n_total-1].params)

for img_idx in range(n_total):
    write_str = "structure " + str(img_idx) + ": " + str(structures[img_idx].params) + "\n"
    f_log.write(write_str)

f_log.write("\n")

updated_structures = deepcopy(structures)
energies = np.zeros(n_intermediate)
energy_errors = np.zeros(n_intermediate)
parameters = np.zeros((n_intermediate, len(structures[0].params)))
parameter_errors = np.zeros((n_intermediate, len(structures[0].params)))

# lsi = [[None] * n_intermediate] * n_iter
lsi = obj()
update_params = obj()
mean_avg = obj()
upper_avg = obj()
lower_avg = obj()
ener_avg = obj()
ener_err_avg = obj()

# Loop over force-free pathway identification steps
for iter_i in range(n_iter):
    update_params[iter_i] = obj()
    lsi[iter_i] = obj()
    mean_avg[iter_i] = obj()
    upper_avg[iter_i] = obj()
    lower_avg[iter_i] = obj()
    ener_avg[iter_i] = obj()
    ener_err_avg[iter_i] = obj()

    if iter_i == 0:
        shift_params = [init_params_shift_factor] * (len(structures[0].params) - 1)
    else:
        shift_params = [0.] * (len(structures[0].params) - 1)
    f_log.write(f"========== ITER {iter_i+1} ==========\n")
    
    # Loop over intermediate structure to perform subspace line search on each
    for int_str in range(n_intermediate):
        true_idx = int_str + 1
        f_log.write(f"-------- intermediate image {true_idx} --------\n")
    
        prev_str = updated_structures[true_idx-1].params
        next_str = updated_structures[true_idx+1].params
        curr_str = updated_structures[true_idx].params
        
        write_str = "initially, prev_str: " + str(prev_str) + "\n"
        f_log.write(write_str)
        write_str = "initially, curr_str: " + str(curr_str) + "\n"
        f_log.write(write_str)
        write_str = "initially, next_str: " + str(next_str) + "\n\n"
        f_log.write(write_str)
    
        prev_state = backward(prev_str)
        next_state = backward(next_str)
        curr_state = backward(curr_str)
    
        replacements = {
    	'initial_state': start_state,
    	'final_state': end_state,
    	'previous_state': prev_state,
    	'next_state': next_state,
    	'num_images': n_total,
    	'spring_const': spring_const
    	}
        pes_springs_template_fname = 'pes_springs_template.py'
        with open(pes_springs_template_fname, 'r') as f:
            content = f.readlines()
        new_content = []
        for line in content:
            for key, value in replacements.items():
                if line.startswith(key):
                    line = f"{key} = {convert_to_string(value)}\n"
            new_content.append(line)
        with open(pes_springs_template_fname, 'w') as f:
            f.writelines(new_content)
    
        tangent_vector = get_tangent(prev_str, next_str, curr_str, true_idx)
        orthogonal_subspace = orthonormal_subspace.orthogonal_subspace_basis(tangent_vector)
        
        write_str = "tangent vector: " + str(tangent_vector) + "\n\n"
        f_log.write(write_str)
    
        subspace_vectors_in_parameter_space = orthogonal_subspace
        for i_v, v in enumerate(subspace_vectors_in_parameter_space):
            write_str = "subspace vector " + str(i_v+1) + ": " + str(v) + "\n"
            f_log.write(write_str)
            assert abs(np.dot(v, tangent_vector)) < 1e-8, "ERROR: subspace vector not orthogonal to tangent"
        f_log.write("\n")
        
        fwd_sub = partial(forward_subspace, curr_str, subspace_vectors_in_parameter_space)
        bwd_sub = partial(backward_subspace, curr_str, subspace_vectors_in_parameter_space)
        
        write_str = "fwd_sub(curr_state): " + str(fwd_sub(curr_state)) + " should be ~0\n"
        f_log.write(write_str)
        write_str = "forward(bwd_sub(fwd_sub(curr_state))): " + str(forward(bwd_sub(fwd_sub(curr_state)))) + " should be the same as " + str(curr_str) + "\n\n"
        f_log.write(write_str)
        
        subspace_curr_str = ParameterStructure(
            forward = fwd_sub,
            backward = bwd_sub,
            params = fwd_sub(curr_state),
            elem = ['C'] + 3*['H'] + 2*['F'],# + ['Cl'],
            units = 'A',
            tol = 1e-5
            )
        subspace_hessian_path_name = f'transition_pathway/iter_{iter_i+1}/intermediate_image_{true_idx}/subspace/true_hessian'
        subspace_hessian = ParameterHessian(structure = subspace_curr_str)
        subspace_hessian.compute_fdiff(
            mode      = 'nexus',
            path      = subspace_hessian_path_name,
            dp        = [0.01] * len(subspace_vectors_in_parameter_space), # without scaling, they are unit vectors
            pes_func  = get_pes_job,
            load_func = load_E_err
        )
        subspace_directions = np.array(subspace_hessian.get_directions())
        subspace_hessian_eigenvalues = np.array(subspace_hessian.Lambda)
        
        write_str = "subspace Hessian eigenvalues: " + str(subspace_hessian_eigenvalues) + "\n"
        f_log.write(write_str)
        write_str = "subspace Hessian eigenvectors: \n" + str(subspace_directions) + "\n\n"
        f_log.write(write_str)
    
        surrogate_path_name = f'transition_pathway/iter_{iter_i+1}/intermediate_image_{true_idx}/subspace/surrogate'
        surrogate = generate_surrogate(
            path = surrogate_path_name,
            mode = 'nexus',
            # pes_func = get_pes_springs_job,
            pes_func = get_pes_job,
            load_func = load_E_err,
            structure = subspace_hessian.structure,
            hessian = subspace_hessian,
            fit_kind = poly_fit_order,
            window_frac = window_frac, #GRI window_frac means maximum displacement in units of lambda of each direction
            M = 17,
            fname = 'surrogate.p',
        )
        
        surrogate.run_jobs(interactive=False)
        surrogate.load_results(set_target=True)
        
        for i in range(len(subspace_hessian_eigenvalues)):
            surrogate.ls(i).target_x0 = surrogate.ls(i).compute_bias_of(W = surrogate.ls(i).Lambda * 0.01)[0][0]
    
        # optimize line-search parameters to given tolerances
        subspace_epsilon_p = []
        for i_sv, sv in enumerate(subspace_vectors_in_parameter_space):
            subspace_epsilon_p.append(np.absolute(np.dot(np.array(epsilon_p), sv)))
        
        # if iter_i == 0:
        #     shift_params = []
        #     for i_sv, sv in enumerate(subspace_vectors_in_parameter_space):
        #         shift_params.append(np.absolute(np.dot(np.array(init_params_shift), sv)))
        # else:
        #     shift_params = [0.0] * len(subspace_epsilon_p)
        
        write_str = "subspace epsilon_p: " + str(subspace_epsilon_p) + ",\n"
        f_log.write(write_str)
        write_str = "corresponding to parameter space epsilon_p: " + str(epsilon_p) + "\n\n"
        
        surrogate.optimize(
            N = 500,
            # kind = 'thermal',
            # temperature = 0.005, # 0.0025 was too small, 0.005 worked
            kind = 'ls',
            M = 7,
            fit_kind = poly_fit_order,
            epsilon_p = subspace_epsilon_p,
            reoptimize = False,
        )
    
        surrogate.write_to_disk('surrogate.p', overwrite = False)
        surrogate_diagnostics(surrogate) #GRI this makes it crash
        plot_surrogate_pes(surrogate)
        surrogate_pes_fname = f'pes_plots/surrogate_pes_iter_{iter_i+1}_intermediate_image_{true_idx}.pdf'
        plt.savefig(surrogate_pes_fname)
    
        ls_path_name = f'transition_pathway/iter_{iter_i+1}/intermediate_image_{true_idx}/subspace/{ls_dirname}'
        
        #GRI USE THIS SECTION FOR NOISELESS AND NOISY DFT SEARCH
        if not run_dmc:
            lsi[iter_i][int_str] = LineSearchIteration(
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
                lsi[iter_i][int_str].pls(i).run_jobs(interactive=False)
                lsi[iter_i][int_str].pls(i).load_results(add_sigma=add_sigma)
                lsi[iter_i][int_str].propagate(i)
        
            # Add the final eqm structure
            lsi[iter_i][int_str].pls(n_ls_iter).run_jobs(interactive=False, eqm_only=True)
            lsi[iter_i][int_str].pls(n_ls_iter).load_eqm_results(add_sigma=add_sigma)
        #GRI END SECTION
        
        #GRI USE THIS SECTION FOR DMC SEARCH
        else:
            var_eff = get_var_eff(
                surrogate.structure,
                partial(dmc_pes_job, 1),
                path = f'transition_pathway/iter_{iter_i+1}/intermediate_image_{true_idx}/subspace/dmc/test',
                suffix = '/dmc/dmc.in.xml',
            )
            
            lsi[iter_i][int_str] = LineSearchIteration(
                path = ls_path_name,
                mode = 'nexus',
                surrogate = surrogate,
                pes_func = partial(dmc_pes_job, dmc_enhancement_factor),
                pes_args = {'var_eff': var_eff}, # supply var_eff to target right sigma
                load_func = nexus_qmcpack_analyzer,
                shift_params = shift_params,
                load_args = {'suffix': '/dmc/dmc.in.xml' } # point to right file
            )
            
            for i in range(n_ls_iter):
                lsi[iter_i][int_str].pls(i).run_jobs(interactive=False)
                lsi[iter_i][int_str].pls(i).load_results()
                lsi[iter_i][int_str].propagate(i)
            
            # Add the final eqm structure
            lsi[iter_i][int_str].pls(n_ls_iter).run_jobs(interactive=False, eqm_only=True)
            lsi[iter_i][int_str].pls(n_ls_iter).load_eqm_results()
        #GRI END SECTION
        
        # loop over parameters (and use loop index to get error bars)
        count = 0.
        mean_sum = np.zeros_like(np.array(structures[0].params))
        upper_squared_sum = np.zeros_like(np.array(structures[0].params))
        lower_squared_sum = np.zeros_like(np.array(structures[0].params))
        subspace_mean_sum = np.zeros_like(lsi[iter_i][int_str].get_params()[0][0])
        subspace_squared_err_sum = np.zeros_like(lsi[iter_i][int_str].get_params()[0][0])
        ener_sum = 0.
        ener_err_squared_sum = 0.
        
        for i_p, p in enumerate(lsi[iter_i][int_str].get_params()[0]):
            p_original_param_space = forward(bwd_sub(p)).tolist()
            
            # get error for parameter p in subspace and determine structural upper and lower bounds
            err_p = lsi[iter_i][int_str].get_params()[1][i_p]
            p_upper = p + err_p
            p_lower = p - err_p
            
            # now calculate the difference between the upper bound (in the original parameter space) and the mean (in the original parameter space)
            # and repeat for lower bound
            # then take the absolute value to get +ve and -ve errors on structural parameter estimates
            dp_upper_original_param_space = np.absolute((np.array(forward(bwd_sub(p_upper))) - np.array(p_original_param_space))).tolist()
            dp_lower_original_param_space = np.absolute((np.array(forward(bwd_sub(p_lower))) - np.array(p_original_param_space))).tolist()
            
            write_str = "ls iter " + str(i_p) + ":\n    " + str(p_original_param_space) + " (in subspace: " + str(p) + ")\n"
            f_log.write(write_str)
            write_str = "    + " + str(dp_upper_original_param_space) + " / - " + str(dp_lower_original_param_space) + \
                        " (in subspace: +/- " + str(err_p) + ")\n"
            f_log.write(write_str)
            write_str = "    energy: " + str(lsi[iter_i][int_str].pls_list[i_p].structure.value) + " +/- " + str(lsi[iter_i][int_str].pls_list[i_p].structure.error) + " eV\n"
            f_log.write(write_str)
            
            if i_p >= n_warmup:
                count += 1.
                mean_sum += np.array(p_original_param_space)
                upper_squared_sum += np.square(dp_upper_original_param_space)
                lower_squared_sum += np.square(dp_lower_original_param_space)
                subspace_mean_sum += np.array(p)
                subspace_squared_err_sum += np.square(err_p)
                ener_sum += lsi[iter_i][int_str].pls_list[i_p].structure.value
                ener_err_squared_sum += (lsi[iter_i][int_str].pls_list[i_p].structure.error) ** 2.
        
        mean_avg[iter_i][int_str] = mean_sum / count
        upper_avg[iter_i][int_str] = np.sqrt(upper_squared_sum / count)
        lower_avg[iter_i][int_str] = np.sqrt(lower_squared_sum / count)
        subspace_mean_avg = subspace_mean_sum / count
        subspace_err_avg = np.sqrt(subspace_squared_err_sum / count)
        ener_avg[iter_i][int_str] = ener_sum / count
        ener_err_avg[iter_i][int_str] = (ener_err_squared_sum / count) ** 0.5
            
        f_log.write("\n")
        
        write_str = "mean over last " + str(n_ls_iter - n_warmup + 1) + " iterations:\n    " + str(mean_avg[iter_i][int_str]) + " (in subspace: " \
                    + str(subspace_mean_avg) + ")\n"
        f_log.write(write_str)
        write_str = "    + " + str(upper_avg[iter_i][int_str]) + " / - " + str(lower_avg[iter_i][int_str]) + \
                    " (in subspace: +/- " + str(subspace_err_avg) + ")\n"
        f_log.write(write_str)
        write_str = "    energy: " + str(ener_avg[iter_i][int_str]) + " +/- " + str(ener_err_avg[iter_i][int_str]) + " eV\n"
        f_log.write(write_str)
        
        # update_params[iter_i][int_str] = forward(bwd_sub(lsi[iter_i][int_str].get_params()[0][-1]))
        update_params[iter_i][int_str] = forward(bwd_sub(subspace_mean_avg))
        
        lsi[iter_i][int_str].plot_convergence()
        
        convergence_fname = f'{convergence_dirname}/stochastic_parameter_convergence_iter_{iter_i+1}_intermediate_image_{true_idx}.pdf'
        plt.savefig(convergence_fname)
        
    # Note: This is not clubbed with the inner loop above because updates need to take place after one iteration over all images is over
    for int_str in range(n_intermediate):
        true_idx = int_str + 1
        updated_structures[true_idx] = ParameterStructure(
        forward = forward,
        backward = backward,
        params = update_params[iter_i][int_str],
        elem = ['C'] + 3*['H'] + 2*['F'],# + ['Cl'],
        units = 'A',
        tol = 1e-5
        )
    
    f_log.write("\n************************************\n")
    for img_idx in range(n_total):
        write_str = str(updated_structures[img_idx].params) + "\n"
        f_log.write(write_str)
    f_log.write("************************************\n")

f_log.close()
