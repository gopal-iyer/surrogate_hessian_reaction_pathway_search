import numpy as np
import matplotlib.pyplot as plt

neb_dft_fmax0_01 = np.array([
                    [1.09383588, 1.21812275, 2.575     , 1.434     ],
                    [1.09404329, 1.34361441, 2.2825867 , 1.50870488],
                    [1.08139495, 1.46056362, 2.06231758, 1.67175899],
                    [1.07556845, 1.57079633, 1.86319157, 1.86319157],
                    [1.08139495, 1.68102904, 1.67175899, 2.06231758],
                    [1.09404329, 1.79797824, 1.50870488, 2.2825867 ],
                    [1.09383588, 1.9234699 , 1.434     , 2.575     ],
])

displaced_initial_guess_fmax0_01_d0_05 = np.array([
                    [1.09383588, 1.21812275, 2.575     , 1.434     ],
                    [1.1439982976805558, 1.3904344223908183, 2.326541029747749, 1.5383764168066623],
                    [1.1313853557093765, 1.5073652469371968, 2.109674025356671, 1.6955903500705587],
                    [1.1255684469346252, 1.617233713517085, 1.9116535698564943, 1.8854402482655956],
                    [1.131385355709376, 1.7271046555710516, 1.721278025046155, 2.082953995646968],
                    [1.143998297680556, 1.8431336414802313, 1.5617287982338959, 2.294994394661395],
                    [1.09383588, 1.9234699 , 1.434     , 2.575     ],
])

subspace_dft_fmax0_01_d0_05 = np.array([
                    [1.09383588, 1.21812275, 2.575     , 1.434     ],
                    [1.08552564, 1.29549231, 2.26201257, 1.51476582],
                    [1.08049201, 1.424513  , 2.03105158, 1.65996801],
                    [1.07811936, 1.57201125, 1.83799844, 1.83731264],
                    [1.08047373, 1.71589827, 1.66025801, 2.03198491],
                    [1.08551277, 1.84689806, 1.51535639, 2.26202694],
                    [1.09383588, 1.9234699 , 1.434     , 2.575     ],
])

subspace_dmc_fmax0_01_d0_00 = np.array([
                    [1.09383588, 1.21812275, 2.575     , 1.434     ],
                    [1.07273386, 1.30321101, 2.24039707, 1.47903708],
                    [1.06798403, 1.43513275, 2.0071687 , 1.62862944],
                    [1.06592464, 1.57179542, 1.810473  , 1.80990902],
                    [1.06807542, 1.70436224, 1.62681788, 2.00658569],
                    [1.07405862, 1.82782747, 1.47487547, 2.24284397],
                    [1.09383588, 1.9234699 , 1.434     , 2.575     ],
])

errors_subspace_dmc_fmax0_01_d0_00 = np.array([
                    [1e-10, 1e-10, 1e-10, 1e-10],
                    [0.00142011, 0.00515674, 0.00683717, 0.00564778],
                    [0.00130793, 0.00558501, 0.00687262, 0.0040202 ],
                    [0.00165013, 0.00695087, 0.0095288 , 0.00572992],
                    [0.0031074,  0.01046325, 0.01388529, 0.0075065 ],
                    [0.00205385, 0.0072314,  0.00967731, 0.00322303],
                    [1e-10, 1e-10, 1e-10, 1e-10],
])

neb_ccsd_fmax0_01 = np.array([
                    [1.09383588, 1.21812275, 2.575     , 1.434     ],
                    [1.07646877, 1.35925988, 2.28216776, 1.47572529],
                    [1.06634227, 1.47362496, 2.05614345, 1.6434119 ],
                    [1.06300821, 1.57079633, 1.84876688, 1.84876688],
                    [1.06634227, 1.6679677 , 1.6434119 , 2.05614345],
                    [1.07646877, 1.78233278, 1.47572529, 2.28216776],
                    [1.09383588, 1.9234699 , 1.434     , 2.575     ],
])

img_idx = np.array([1, 2, 3, 4, 5, 6, 7])

'''
# begin GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

# kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
# kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)

X = np.array([img_idx]).T
y0 = subspace_dmc_fmax0_01_d0_00[:,0]
y1 = subspace_dmc_fmax0_01_d0_00[:,1]
y2 = subspace_dmc_fmax0_01_d0_00[:,2]
y3 = subspace_dmc_fmax0_01_d0_00[:,3]

y0_scaler = StandardScaler()
y1_scaler = StandardScaler()
y2_scaler = StandardScaler()
y3_scaler = StandardScaler()
# Scale the target data (y)
y0_scaled = y0_scaler.fit_transform(y0.reshape(-1, 1)).ravel()
y1_scaled = y1_scaler.fit_transform(y1.reshape(-1, 1)).ravel()
y2_scaled = y2_scaler.fit_transform(y2.reshape(-1, 1)).ravel()
y3_scaled = y3_scaler.fit_transform(y3.reshape(-1, 1)).ravel()
dy0 = errors_subspace_dmc_fmax0_01_d0_00[:,0]
dy1 = errors_subspace_dmc_fmax0_01_d0_00[:,1]
dy2 = errors_subspace_dmc_fmax0_01_d0_00[:,2]
dy3 = errors_subspace_dmc_fmax0_01_d0_00[:,3]
dy0_scaled = dy0 / y0_scaler.scale_
dy1_scaled = dy1 / y1_scaler.scale_
dy2_scaled = dy2 / y2_scaler.scale_
dy3_scaled = dy3 / y3_scaler.scale_

gp0 = GaussianProcessRegressor(kernel=kernel, alpha=dy0_scaled**2., n_restarts_optimizer=100, random_state=np.random.RandomState(42))
gp1 = GaussianProcessRegressor(kernel=kernel, alpha=dy1_scaled**2., n_restarts_optimizer=100, random_state=np.random.RandomState(42))
gp2 = GaussianProcessRegressor(kernel=kernel, alpha=dy2_scaled**2., n_restarts_optimizer=100, random_state=np.random.RandomState(42))
gp3 = GaussianProcessRegressor(kernel=kernel, alpha=dy3_scaled**2., n_restarts_optimizer=100, random_state=np.random.RandomState(42))

gp0.fit(X, y0_scaled)
gp1.fit(X, y1_scaled)
gp2.fit(X, y2_scaled)
gp3.fit(X, y3_scaled)

X_pred = np.linspace(1, 7, 1000).reshape(-1, 1)

y0_pred_scaled, sigma0_scaled = gp0.predict(X_pred, return_std=True)
y1_pred_scaled, sigma1_scaled = gp1.predict(X_pred, return_std=True)
y2_pred_scaled, sigma2_scaled = gp2.predict(X_pred, return_std=True)
y3_pred_scaled, sigma3_scaled = gp3.predict(X_pred, return_std=True)
y0_pred = y0_scaler.inverse_transform(y0_pred_scaled.reshape(-1, 1)).ravel()
y1_pred = y1_scaler.inverse_transform(y1_pred_scaled.reshape(-1, 1)).ravel()
y2_pred = y2_scaler.inverse_transform(y2_pred_scaled.reshape(-1, 1)).ravel()
y3_pred = y3_scaler.inverse_transform(y3_pred_scaled.reshape(-1, 1)).ravel()
sigma0 = sigma0_scaled * y0_scaler.scale_  # Rescale the standard deviation
sigma1 = sigma1_scaled * y1_scaler.scale_  # Rescale the standard deviation
sigma2 = sigma2_scaled * y2_scaler.scale_  # Rescale the standard deviation
sigma3 = sigma3_scaled * y3_scaler.scale_  # Rescale the standard deviation
# end GPR
'''

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 12))

axisfontsize = 20
labelfontsize = 20

# #### SECTION 1 ####
# axs[0].plot(img_idx, neb_dft_fmax0_01[:,0], marker='^', color='g', linewidth=0.9, label='DFT-NEB')
# axs[0].plot(img_idx, displaced_initial_guess_fmax0_01_d0_05[:,0], marker='o', markerfacecolor='none', color='r', linestyle='--', linewidth=0.9, label='Displaced DFT guess')
# axs[0].plot(img_idx, subspace_dft_fmax0_01_d0_05[:,0], marker='o', color='r', linewidth=0.9, label='Subspace optimization (DFT)')
# axs[0].errorbar(img_idx, subspace_dmc_fmax0_01_d0_00[:,0], yerr=errors_subspace_dmc_fmax0_01_d0_00[:,0], marker='s', color='b', capsize=2.5, elinewidth=0.9, linewidth=0.9, label='Subspace optimization (DMC)')
# axs[0].plot(img_idx, neb_ccsd_fmax0_01[:,0], marker='*', color='k', linewidth=0.9, label='CCSD-NEB')
# # axs[0].plot(X_pred.ravel(), y0_pred, color='darkorchid', label='GPR (DMC)')
# # axs[0].fill_between(X_pred.ravel(), y0_pred - sigma0, y0_pred + sigma0, alpha=0.5, color='darkorchid', label='GPR (SD)')
# axs[0].set_ylabel('p$_0$ (Å)', fontsize=axisfontsize)
#
# #### SECTION 2 ####
# axs[1].plot(img_idx, neb_dft_fmax0_01[:,1], marker='^', color='g', linewidth=0.9, label='DFT-NEB')
# axs[1].plot(img_idx, displaced_initial_guess_fmax0_01_d0_05[:,1], marker='o', markerfacecolor='none', color='r', linestyle='--', linewidth=0.9, label='Displaced DFT guess')
# axs[1].plot(img_idx, subspace_dft_fmax0_01_d0_05[:,1], marker='o', color='r', linewidth=0.9, label='Subspace optimization (DFT)')
# axs[1].errorbar(img_idx, subspace_dmc_fmax0_01_d0_00[:,1], yerr=errors_subspace_dmc_fmax0_01_d0_00[:,1], marker='s', color='b', capsize=2.5, elinewidth=0.9, linewidth=0.9, label='Subspace optimization (DMC)')
# axs[1].plot(img_idx, neb_ccsd_fmax0_01[:,1], marker='*', color='k', linewidth=0.9, label='CCSD-NEB')
# # axs[1].plot(X_pred.ravel(), y1_pred, color='darkorchid', label='GPR (DMC)')
# # axs[1].fill_between(X_pred.ravel(), y1_pred - sigma1, y1_pred + sigma1, alpha=0.5, color='darkorchid', label='GPR (SD)')
# axs[1].set_ylabel('p$_1$ (rad)', fontsize=axisfontsize)

#### SECTION 3 ####
axs[0].plot(img_idx, neb_dft_fmax0_01[:,2], marker='^', color='g', linewidth=0.9, label='DFT-NEB')
axs[0].plot(img_idx, displaced_initial_guess_fmax0_01_d0_05[:,2], marker='o', markerfacecolor='none', color='r', linestyle='--', linewidth=0.9, label='Displaced DFT guess')
axs[0].plot(img_idx, subspace_dft_fmax0_01_d0_05[:,2], marker='o', color='r', linewidth=0.9, label='Subspace optimization (DFT)')
axs[0].errorbar(img_idx, subspace_dmc_fmax0_01_d0_00[:,2], yerr=errors_subspace_dmc_fmax0_01_d0_00[:,2], marker='s', color='b', capsize=2.5, elinewidth=0.9, linewidth=0.9, label='Subspace optimization (DMC)')
axs[0].plot(img_idx, neb_ccsd_fmax0_01[:,2], marker='*', color='k', linewidth=0.9, label='CCSD-NEB')
# axs[0].plot(X_pred.ravel(), y2_pred, color='darkorchid', label='GPR (DMC)')
# axs[0].fill_between(X_pred.ravel(), y2_pred - sigma2, y2_pred + sigma2, alpha=0.5, color='darkorchid', label='GPR (SD)')
axs[0].set_ylabel('p$_2$ (Å)', fontsize=axisfontsize)

#### SECTION 4 ####
axs[1].plot(img_idx, neb_dft_fmax0_01[:,3], marker='^', color='g', linewidth=0.9, label='DFT-NEB')
axs[1].plot(img_idx, displaced_initial_guess_fmax0_01_d0_05[:,3], marker='o', markerfacecolor='none', color='r', linestyle='--', linewidth=0.9, label='Displaced DFT guess')
axs[1].plot(img_idx, subspace_dft_fmax0_01_d0_05[:,3], marker='o', color='r', linewidth=0.9, label='Subspace optimization (DFT)')
axs[1].errorbar(img_idx, subspace_dmc_fmax0_01_d0_00[:,3], yerr=errors_subspace_dmc_fmax0_01_d0_00[:,3], marker='s', color='b', capsize=2.5, elinewidth=0.9, linewidth=0.9, label='Subspace optimization (DMC)')
axs[1].plot(img_idx, neb_ccsd_fmax0_01[:,3], marker='*', color='k', linewidth=0.9, label='CCSD-NEB')
# axs[1].plot(X_pred.ravel(), y3_pred, color='darkorchid', label='GPR (DMC)')
# axs[1].fill_between(X_pred.ravel(), y3_pred - sigma3, y3_pred + sigma3, alpha=0.5, color='darkorchid', label='GPR (SD)')
axs[1].set_ylabel('p$_3$ (Å)', fontsize=axisfontsize)

axs[1].set_xlabel('Path image', fontsize=axisfontsize)
axs[1].set_xticks(ticks=[1,2,3,4,5,6,7])
axs[1].set_xticklabels(labels=['0\n(initial)', '1', '2', '3', '4', '5', '6\n(final)'])

axs[0].legend(loc='upper right', fontsize=15)

for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=labelfontsize)

# handles, labels = [], []
#
# for handle, label in zip(*axs[0].get_legend_handles_labels()):
#     handles.append(handle)
#     labels.append(label)
#
# # Place the legend to the right of the last subplot
# fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1))

# Adjust the layout to make room for the legend
# plt.tight_layout()

plt.savefig('p23_vs_rc_GPR.png')
