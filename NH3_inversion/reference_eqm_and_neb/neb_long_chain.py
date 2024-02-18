#! /usr/bin/env python
##! /usr/bin/env python3

##SBATCH --time=48:00:00
##SBATCH --nodes=1
##SBATCH --partition=bigmem
##SBATCH --ntasks-per-node=24
##SBATCH --mem-per-cpu=64G # max memory per core
##SBATCH --mem=180G
##SBATCH --exclusive
##SBATCH --account=brubenst-condo
##SBATCH --constraint=broadwell
##SBATCH -o neb.out
##SBATCH -e neb.err

from ase.neb import NEB, interpolate
from ase import io
from ase.optimize import BFGS
from ase import Atoms
#from ase.build import molecule, bulk, fcc111, add_adsorbate
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.constraints import FixAtoms
from ase.visualize import view
#from ase.calculators.espresso import Espresso
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from pyscf import df, scf, dft, grad
from pyscf import gto as gto_loc

initial = io.read('init_fixed.xyz', parallel=False)
final = io.read('final_fixed.xyz', parallel=False)

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

images = [initial]
for index in range(15):
    calc = PySCF()
    image = initial.copy()
    image.set_calculator(calc)
    images.append(image)

images.append(final)

interpolate(images)

for image in images:
	c = FixAtoms(indices=[0])
	image.set_constraint(c)

neb = NEB(images)
opt = BFGS(neb, trajectory='NEB_long_chain_f0.01.traj', logfile='NEB_long_chain_f0.01.log')
opt.run(fmax=0.01)
