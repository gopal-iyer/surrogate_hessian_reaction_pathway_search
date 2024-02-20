import numpy as np
import matplotlib.pyplot as plt

neb_dft_fmax0_01 = np.array([
                    [-11.7263372163866],
                    [-11.7219838978909],
                    [-11.7187506806269],
                    [-11.721983897989],
                    [-11.7263372163866],
])

subspace_dft_fmax0_01_d0_05 = np.array([
                    [-11.7263372163866],
                    [-11.7219711127528],
                    [-11.7187553460208],
                    [-11.7220404588077],
                    [-11.7263372163866],
])

subspace_dmc_fmax0_01_d0_00 = np.array([
                    [-11.729313848850266],
                    [-11.7253845132507],
                    [-11.722346061988105],
                    [-11.724770664645002],
                    [-11.729519085895],
])

errors_subspace_dmc_fmax0_01_d0_00 = np.array([
                    [0.000387390058075996],
                    [0.001524262904921619],
                    [0.0011458122529824078],
                    [0.001096311985667188],
                    [0.0006515890433992935],
])

neb_ccsd_fmax0_01 = np.array([
                    [-11.7075809558418],
                    [-11.702762500978],
                    [-11.6986962693131],
                    [-11.7027625007496],
                    [-11.7075809558418],
])

img_idx = np.array([1, 2, 3, 4, 5])

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
dy_scaled = dy_scaled = dy / y_scaler.scale_

gp = GaussianProcessRegressor(kernel=kernel, alpha=dy_scaled**2., n_restarts_optimizer=100, random_state=np.random.RandomState(42))

gp.fit(X, y_scaled)

loss = gp.log_marginal_likelihood_value_
# print("loss = ", loss)

X_pred = np.linspace(1, 5, 1000).reshape(-1, 1)

y_pred_scaled, sigma_scaled = gp.predict(X_pred, return_std=True)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
sigma = sigma_scaled * y_scaler.scale_  # Rescale the standard deviation

plt.plot(X_pred.ravel(), y_pred, color='darkorchid')
plt.fill_between(X_pred.ravel(), y_pred - sigma, y_pred + sigma, alpha=0.5, color='darkorchid', label='GPR fit to DMC')
# end GPR

plt.xlabel('Path image', fontsize=16)
plt.ylabel('E$-$E$_0$ (eV)', fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=12)

plt.xticks(ticks=[1,2,3,4,5,], labels=['0\n(initial)', '1', '2', '3', '4\n(final)'], fontsize=12)
plt.legend(loc=0, fontsize='large')

handles, labels = plt.gca().get_legend_handles_labels()
legend_order = [0,1,2,4,3]
plt.legend([handles[idx] for idx in legend_order],[labels[idx] for idx in legend_order], fontsize='large')

plt.show()
