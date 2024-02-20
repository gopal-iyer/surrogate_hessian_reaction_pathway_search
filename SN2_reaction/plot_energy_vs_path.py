import numpy as np
import matplotlib.pyplot as plt

neb_dft_fmax0_01 = np.array([
                    [-56.09499079580641],
                    [-56.09346438366686],
                    [-56.08908711133993],
                    [-56.086729588942994],
                    [-56.089087111549304],
                    [-56.09346438350624],
                    [-56.09499079580876],
])

subspace_dft_fmax0_01_d0_05 = np.array([
                    [-56.09499079580641],
                    [-56.09468345638879],
                    [-56.08971901989591],
                    [-56.08688734934785],
                    [-56.08969441297964],
                    [-56.09464319116469],
                    [-56.09499079580876],
])

subspace_dmc_fmax0_01_d0_00 = np.array([
                    [-56.09227382685],
                    [-56.08809155053839],
                    [-56.07661826563116],
                    [-56.069943871364806],
                    [-56.07625553912195],
                    [-56.08820916600269],
                    [-56.09231644716216],
])

errors_subspace_dmc_fmax0_01_d0_00 = np.array([
                    [0.00022799142173683844],
                    [0.00020504764811906683],
                    [0.0004107763190231726],
                    [0.0007035518260526389],
                    [0.0013030772610916855],
                    [0.0006849436540917031],
                    [0.00015522425388810328],
])

neb_ccsd_fmax0_01 = np.array([
                    [-55.9845702397076],
                    [-55.9801244765022],
                    [-55.96898621297992],
                    [-55.96282370667933],
                    [-55.96898621314951],
                    [-55.98012447546988],
                    [-55.9845702377874],
])

img_idx = np.array([1, 2, 3, 4, 5, 6, 7])

plt.plot(img_idx, neb_dft_fmax0_01[:,0]-neb_dft_fmax0_01[0,0], marker='^', color='g', linewidth=0.9, label='DFT-NEB')
plt.plot(img_idx, subspace_dft_fmax0_01_d0_05[:,0]-subspace_dft_fmax0_01_d0_05[0,0], marker='o', color='r', linewidth=0.9, label='Subspace optimization (DFT)')
plt.errorbar(img_idx, subspace_dmc_fmax0_01_d0_00[:,0]-subspace_dmc_fmax0_01_d0_00[0,0], yerr=errors_subspace_dmc_fmax0_01_d0_00[:,0], marker='s', color='b', capsize=2.5, elinewidth=0.9, linewidth=0.9, label='Subspace optimization (DMC)')
plt.plot(img_idx, neb_ccsd_fmax0_01[:,0]-neb_ccsd_fmax0_01[0,0], marker='*', color='k', linewidth=0.9, label='CCSD-NEB')

# begin GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

# kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
# kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)

X = np.array([img_idx]).T
y = subspace_dmc_fmax0_01_d0_00[:,0] - subspace_dmc_fmax0_01_d0_00[0,0]

y_scaler = StandardScaler()
# Scale the target data (y)
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
dy = errors_subspace_dmc_fmax0_01_d0_00[:,0]
dy_scaled = dy / y_scaler.scale_

gp = GaussianProcessRegressor(kernel=kernel, alpha=dy_scaled**2., n_restarts_optimizer=20, random_state=np.random.RandomState(42))

gp.fit(X, y_scaled)

X_pred = np.linspace(1, 7, 1000).reshape(-1, 1)

y_pred_scaled, sigma_scaled = gp.predict(X_pred, return_std=True)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
sigma = sigma_scaled * y_scaler.scale_  # Rescale the standard deviation

plt.plot(X_pred.ravel(), y_pred, color='darkorchid')
plt.fill_between(X_pred.ravel(), y_pred - sigma, y_pred + sigma, alpha=0.5, color='darkorchid', label='GPR fit to DMC')
# end GPR

plt.xlabel('Path image', fontsize=16)
plt.ylabel('E$-$E$_0$ (eV)', fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=12)

plt.xticks(ticks=[1,2,3,4,5,6,7], labels=['0\n(initial)', '1', '2', '3', '4', '5', '6\n(final)'])

handles, labels = plt.gca().get_legend_handles_labels()
legend_order = [0,1,2,4,3]
plt.legend([handles[idx] for idx in legend_order],[labels[idx] for idx in legend_order], fontsize='large')
# plt.savefig('SN2_E_vs_rc_GPR')
plt.show()
