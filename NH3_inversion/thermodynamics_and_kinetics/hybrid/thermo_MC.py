#! /usr/bin/env python

from ase import io
from ase.thermochemistry import IdealGasThermo
from ase.vibrations import Vibrations
from ase.optimize import BFGS
from ase import Atoms
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.constraints import FixAtoms
from ase.visualize import view
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from pyscf import df, scf, dft, grad
from pyscf import gto as gto_loc
from scipy.optimize import minimize
from math import sqrt, pi
from copy import deepcopy

system = 'NH3'
fname = 'deltaG_init'
mol_vib = io.read('relax_init.xyz', parallel=False)
# mol_vib = io.read('structure_2.xyz', parallel=False)
# mol_vib = io.read('relax_final.xyz', parallel=False)
potentialenergy = -11.72914220170045 # dmc init
delta_pe = 0.0003528634547161434 # dmc init
# potentialenergy = -11.721249498531668 # dmc saddle
# delta_pe = 0.00016746532688090225 # dmc saddle
# potentialenergy = -11.729568948022319 # dmc final
# delta_pe = 0.0007405766275325181 # dmc final
T = 298.15
P = 101325.

mol_vib_mean_params = np.array([1.00641628, 1.94715553])
mol_vib_std_errors = np.array([0.00101238, 0.0051651])
n_samples = int(50 ** len(mol_vib_mean_params))

# bond angle between r0-rc and r1-rc bonds
def bond_angle(r0, rc, r1, units = 'ang'):
    v1 = r0 - rc
    v2 = r1 - rc
    cosang = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
    ang = np.arccos(cosang) * 180 / pi if units == 'ang' else np.arccos(cosang)
    return ang
#end def

def distance(r0, r1):
    r = np.linalg.norm(r0 - r1)
    return r

def mean_distances(pairs):
    rs = []
    for pair in pairs:
        rs.append(distance(pair[0], pair[1]))
    #end for
    return np.array(rs).mean()

def invert_pos(pos0, params, forward = None, tol = 1.0e-7, method = 'BFGS'):
    assert forward is not None, 'Must provide forward mapping'
    def dparams_sum(pos1):
        return sum((params - forward(pos1))**2)
    #end def
    pos1 = minimize(dparams_sum, pos0, tol = tol, method = method).x
    return pos1

# pos,cell given in angstrom
def forward(pos, axes = None):
    N, H0, H1, H2 = tuple(pos.reshape(-1, 3))
    negative_x_axis = deepcopy(N)
    negative_x_axis[0] = -1.
    negative_x_axis[1] =  0.
    negative_x_axis[2] =  0.
    r_NH = mean_distances([(N, H0),(N, H1),(N, H2)])
    a_H_negative_x = np.mean([bond_angle(H0, N, negative_x_axis, units='rad'), bond_angle(H1, N, negative_x_axis, units='rad'),
                            bond_angle(H2, N, negative_x_axis, units='rad')]) #GRI IMPORTANT! changing angle units to rad
    return np.array([r_NH, a_H_negative_x])

def backward(params):
    r_NH, a_H_negative_x = tuple(params)
    def backward_aux(p):
        x0, yz0 = tuple(p)
        N = np.array([0.0, 0.0, 0.0])
        H0 = np.array([ -x0, yz0, 0.0])
        H1 = np.array([ -x0, -yz0 * np.sin(pi / 6), yz0 * np.cos(pi / 6)])
        H2 = np.array([ -x0, -yz0 * np.sin(pi / 6), -yz0 * np.cos(pi / 6)])
        pos = np.array([N, H0, H1, H2])
        return pos  # return parameters
    #end def
    def forward_aux(p):
        return forward(backward_aux(p))
    #end def
    theta = a_H_negative_x # approximation
    x0 = r_NH * np.cos(theta)
    yz0 = r_NH * np.sin(theta)
    p0 = [x0, yz0]
    p = invert_pos(pos0 = p0, params = params, forward = forward_aux, tol = 1e-9)
    return backward_aux(p)

class PySCF(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Get positions and atomic numbers from the Atoms object
        positions = self.atoms.positions
        atomic_numbers = self.atoms.numbers
        
        # Calculate energy and forces here using your custom method
        energy = self.compute_energy(positions, atomic_numbers)
        forces = self.compute_forces(positions, atomic_numbers)
        
        # Set results to be used by ASE
        self.results = {'energy': energy, 'forces': forces}

    def compute_energy(self, positions, atomic_numbers):
        # Implement your custom energy calculation here
        mol = gto_loc.Mole()
        mol.atom = [['N', positions[0]], ['H', positions[1]], ['H', positions[2]], ['H', positions[3]]]
        mol.basis    = 'ccecpccpvtz'
        mol.unit     = 'A'
        mol.ecp      = 'ccecp'
        mol.charge   = 0
        mol.spin     = 0
        mol.symmetry = False
        mol.build()
        mf = dft.RKS(mol)
        mf.xc = 'pbe'
        mf.tol = '1e-10'
        energy = mf.kernel()
        return energy

    def compute_forces(self, positions, atomic_numbers):
        # Implement your custom forces calculation here
        mol = gto_loc.Mole()
        mol.atom = [['N', positions[0]], ['H', positions[1]], ['H', positions[2]], ['H', positions[3]]]
        mol.basis    = 'ccecpccpvtz'
        mol.unit     = 'A'
        mol.ecp      = 'ccecp'
        mol.charge   = 0
        mol.spin     = 0
        mol.symmetry = False
        mol.build()
        mf = dft.RKS(mol)
        mf.xc = 'pbe'
        mf.tol = '1e-10'
        mf.kernel()
        mf_grad = grad.RKS(mf)
        forces = mf_grad.kernel()
        return -forces #GRI force = -ve of gradient

mol_vib.calc = PySCF()

vib = Vibrations(mol_vib)
vib.run()
vib_energies = vib.get_energies()

mol_mean = Atoms(system, backward(mol_vib_mean_params))

random_matrix = np.empty((n_samples, len(mol_vib_mean_params)))

for i in range(len(mol_vib_mean_params)):
    random_matrix[:, i] = np.random.normal(mol_vib_mean_params[i], mol_vib_std_errors[i], n_samples)

S = []
for i_structure_params, structure_params in enumerate(random_matrix):
    mol = Atoms(system, backward(structure_params.tolist()))
    thermo = IdealGasThermo(vib_energies=vib_energies,
                            potentialenergy=potentialenergy,
                            atoms=mol,
                            geometry='nonlinear',
                            symmetrynumber=3, spin=0)
    S.append(thermo.get_entropy(temperature=T, pressure=P))
    
    if i_structure_params % 50 == 0:
        print(i_structure_params, " samples done")

S = np.array(S)
S_mean = np.mean(S)
S_err  = np.std(S)

thermo = IdealGasThermo(vib_energies=vib_energies,
                            potentialenergy=potentialenergy,
                            atoms=mol_mean,
                            geometry='nonlinear',
                            symmetrynumber=3, spin=0)
H = thermo.get_enthalpy(temperature=T)
H_err = delta_pe

G = H - T * S_mean
dG = sqrt(H_err ** 2. + (T*S_err)**2.)

print(f"T*S (total) = {T*S_mean} +/- {T*S_err}")
print(f"subtract non-rotational contributions from this to get the rotational part")

f = open(fname, "w")
s = f"{G} +/- {dG}\n"
f.write(s)
f.close()






