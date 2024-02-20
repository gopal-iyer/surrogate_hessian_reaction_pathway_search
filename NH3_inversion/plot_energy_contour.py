import numpy as np
import matplotlib.pyplot as plt

"""
NOTE: The prefix 'true' here means 'obtained using subspace optimization'
"""

bond_length_grid = np.linspace(0.9, 1.1, 21)[3:-3]
bond_angle_grid = np.linspace(45*np.pi/180., 135*np.pi/180., 21)[3:-3]

# linear interpolation
interp_bond_lengths = np.array([1.0191374, 0.96318134, 0.94379237, 0.96318134, 1.0191374])
interp_bond_angles = np.array([1.95773183, 1.77178406, 1.57079633, 1.3698086 , 1.18386082])

# displaced_initial_bond_lengths_fmax0_02 = np.array([1.0191374, 1.054474411115036, 1.0463477577645082, 1.0544744111150348, 1.0191374])
# displaced_initial_bond_angles_fmax0_02 = np.array([1.95773183, 1.795047944754913, 1.5707963267930392, 1.3465447088344442, 1.18386082])

displaced_initial_bond_lengths_fmax0_01 = np.array([1.0191374, 1.0485095467369554, 1.0474828202410524, 1.0485095467369554, 1.0191374])
displaced_initial_bond_angles_fmax0_01 = np.array([1.95773183, 1.7643737860960593, 1.5707963267948966, 1.3772188674937338,  1.18386082])

# Uses NEB (fmax=0.02) as a starting guess but displaced by 0.05 units along each subspace direction, line search averaged over the last three iterations
# Note that if the displacement is set in parameter space, e.g. (0.05 Å, 0.25 rad), symmetrically identical structures are shifted
# by different amounts when projected into the subspace, so we avoid this for clarity of depiction
# true_bond_lengths_dft_fmax0_02_d0_05 = np.array([1.0191374, 1.00509693, 0.99696491, 1.00431857, 1.0191374])
# true_bond_angles_dft_fmax0_02_d0_05 = np.array([1.95773183, 1.79683017, 1.57079633, 1.34473439, 1.18386082])

true_bond_lengths_dft_fmax0_01_d0_05 = np.array([1.0191374, 1.00013997, 0.99787232, 1.00013997, 1.0191374])
true_bond_angles_dft_fmax0_01_d0_05 = np.array([1.95773183, 1.76463016, 1.57079633, 1.37696249, 1.18386082])

# true_bond_lengths_dft_fmax0_01_d0_00 = np.array([1.0191374, 1.00332309, 0.99917184, 1.00332309, 1.0191374])
# true_bond_angles_dft_fmax0_01_d0_00 = np.array([1.95773183, 1.82914585, 1.57079633, 1.31244681, 1.18386082])

# initial displacement along subspace direction ('d' suffix above) is always set to 0 for DMC

### without enhancement ###
# true_bond_lengths_dmc_fmax0_01 = np.array([1.0191374, 0.98905331, 0.98710611, 0.98412528, 1.0191374])
# true_bond_angles_dmc_fmax0_01 = np.array([1.95773183, 1.76468892, 1.57079633, 1.37687761, 1.18386082])

# upper row is -ve error, lower row is +ve error
# errors_dmc_bond_lengths_fmax0_01 = np.array([[0.0, 4.28596739e-03, 0.00450249, 4.81969861e-03, 0.0],
#                                              [0.0, 4.28596739e-03, 0.00450249, 4.81969861e-03, 0.0]])
# upper row is -ve error, lower row is +ve error
# errors_dmc_bond_angles_fmax0_01 = np.array([[0.0, 2.27170360e-05, 0.0, 2.55459869e-05, 0.0],
#                                             [0.0, 2.27170360e-05, 0.0, 2.55459869e-05, 0.0]])
### end without enhancement ###

### with 16-fold (imgs 1, 2) and 64-fold (img 3) enhancement ###
true_bond_lengths_dmc_fmax0_01 = np.array([1.0191374, 0.98912174, 0.98620234, 0.98914004, 1.0191374])
true_bond_angles_dmc_fmax0_01 = np.array([1.95773183, 1.76468856, 1.57079633, 1.37690419, 1.18386082])

# upper row is -ve error, lower row is +ve error
errors_dmc_bond_lengths_fmax0_01 = np.array([[0.0, 2.38490263e-03, 0.00180094, 1.34103311e-03, 0.0],
                                             [0.0, 2.38490263e-03, 0.00180094, 1.34103311e-03, 0.0]])
# upper row is -ve error, lower row is +ve error
errors_dmc_bond_angles_fmax0_01 = np.array([[0.0, 1.26407678e-05, 0.0, 7.10791627e-06, 0.0],
                                            [0.0, 1.26407678e-05, 0.0, 7.10791627e-06, 0.0]])
### end with 16-fold (imgs 1, 2) and 64-fold (img 3) enhancement ###

# NEB using CCSD with fmax = 0.02 eV/Å, initial conditions are DFT-NEB with fmax = 0.02 eV/Å
neb_ccsd_bond_lengths_fmax0_01 = np.array([1.0191374, 0.99152241, 0.99013452, 0.99152241, 1.0191374])
neb_ccsd_bond_angles_fmax0_01 = np.array([1.95773183, 1.76601079, 1.57079633, 1.37558187, 1.18386082])

# neb0_04_bond_lengths = np.array([1.0191374, 1.00450695, 0.99634776, 1.00450695, 1.0191374])
# neb0_04_bond_angles = np.array([1.95773183, 1.79685146, 1.57079633, 1.3447412, 1.18386082])

# neb0_02_bond_lengths = np.array([1.0191374 , 1.00450695, 0.99634776, 1.00450695, 1.0191374])
# neb0_02_bond_angles = np.array([1.95773183, 1.79685146, 1.57079633, 1.3447412, 1.18386082])

neb0_01_bond_lengths = np.array([1.0191374, 0.99851025, 0.99748282, 0.99851025, 1.0191374])
neb0_01_bond_angles = np.array([1.95773183, 1.7646388, 1.57079633, 1.37695385, 1.18386083])

displaced_ts_guess_bond_lengths = np.array([0.98914004])
errors_displaced_ts_guess_bond_lengths = np.array([0.0])
displaced_ts_guess_bond_angles = np.array([1.37690419])
errors_displaced_ts_guess_bond_angles = np.array([0.0])

displaced_ts_final_bond_lengths = np.array([0.99211341])
errors_displaced_ts_final_bond_lengths = np.array([0.00180153])
displaced_ts_final_bond_angles = np.array([1.57370458])
errors_displaced_ts_final_bond_angles = np.array([0.0341974])

X, Y = np.meshgrid(bond_length_grid, bond_angle_grid)
Z = np.zeros((len(bond_length_grid), len(bond_angle_grid)))

for i_bl, bl in enumerate(bond_length_grid):
    for i_ba, ba in enumerate(bond_angle_grid):
        read_fname = f'ccsd_energy_grid/p0_{bl:.3f}_p1_{ba:.3f}.dat'
        Z[i_bl, i_ba] = np.loadtxt(read_fname)

# fig, ax = plt.subplots()
plt.contourf(X, Y, Z.T, 50)
plt.colorbar()

plt.plot(interp_bond_lengths, interp_bond_angles, marker='o', color='yellow', markerfacecolor='none', linewidth=1.2, linestyle='--', label='Linear interpolation')
# plt.plot(neb0_02_bond_lengths, neb0_02_bond_angles, marker='o', color='yellow', linewidth=1.2, label='DFT-NEB (tol = 0.02 eV/Å)')
# plt.plot(displaced_initial_bond_lengths_fmax0_02, displaced_initial_bond_angles_fmax0_02, marker='s', color='magenta', markerfacecolor='none', linewidth=0.9, linestyle='--', label='Displaced DFT guess')
# plt.plot(true_bond_lengths_dft_fmax0_02_d0_05, true_bond_angles_dft_fmax0_02_d0_05, marker='s', color='magenta', linewidth=0.9, label='Subspace optimization (DFT)')
plt.plot(displaced_initial_bond_lengths_fmax0_01, displaced_initial_bond_angles_fmax0_01, marker='s', color='magenta', markerfacecolor='none', linewidth=1.2, linestyle='--', label='Displaced DFT guess')
plt.plot(true_bond_lengths_dft_fmax0_01_d0_05, true_bond_angles_dft_fmax0_01_d0_05, marker='s', color='magenta', linewidth=1.2, label='Subspace optimization (DFT)')
# plt.plot(true_bond_lengths_dft_fmax0_01_d0_00, true_bond_angles_dft_fmax0_01_d0_00, marker='^', color='violet', linewidth=1.2, label='Subspace optimization (DFT)')
plt.plot(neb_ccsd_bond_lengths_fmax0_01, neb_ccsd_bond_angles_fmax0_01, marker='*', color='white', linewidth=0.9, label='CCSD-NEB')
plt.errorbar(true_bond_lengths_dmc_fmax0_01, true_bond_angles_dmc_fmax0_01, yerr=errors_dmc_bond_angles_fmax0_01, xerr=errors_dmc_bond_lengths_fmax0_01, marker='s', color='cyan', capsize=2.5, linewidth=1.2, elinewidth=0.6, label='Subspace optimization (DMC)')
# plt.plot(neb0_04_bond_lengths, neb0_04_bond_angles, marker='h', color='g', linewidth=0.9, label='DFT-NEB (tol = 0.04 eV/Å)')
plt.plot(neb0_01_bond_lengths, neb0_01_bond_angles, marker='o', color='yellow', linewidth=1.2, label='DFT-NEB')

# plt.errorbar(displaced_ts_guess_bond_lengths, displaced_ts_guess_bond_angles,
#             yerr=errors_displaced_ts_guess_bond_angles, xerr=errors_displaced_ts_guess_bond_lengths,
#             marker='s', markerfacecolor='none', color='r', capsize=2.5, linewidth=1.2, elinewidth=0.6, label='Saddle point guess')
# plt.errorbar(displaced_ts_final_bond_lengths, displaced_ts_final_bond_angles,
#             yerr=errors_displaced_ts_final_bond_angles, xerr=errors_displaced_ts_final_bond_lengths,
#             marker='s', color='r', capsize=2.5, linewidth=1.2, elinewidth=0.6, label='DMC saddle point search')
# plt.arrow(displaced_ts_guess_bond_lengths[0], displaced_ts_guess_bond_angles[0],
#           displaced_ts_final_bond_lengths[0] - displaced_ts_guess_bond_lengths[0],
#           displaced_ts_final_bond_angles[0] - displaced_ts_guess_bond_angles[0],
#           linewidth=0.0, width=0.00025, color='k', zorder=10, head_width=0.002, head_length=0.03, length_includes_head=True)

plt.tick_params(axis='both', which='major', labelsize=15)
plt.annotate("", xy=(0.5, 0.5), xytext=(0, 0), arrowprops=dict(arrowstyle='->'))

plt.legend(loc=0, fontsize='small')
plt.xlabel('p$_0$ (Å)', fontsize=20)
plt.ylabel('p$_1$ (rad)', fontsize=20)
plt.show()
