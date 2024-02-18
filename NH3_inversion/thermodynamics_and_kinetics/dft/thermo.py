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

mol = io.read('relax_init.xyz', parallel=False)
# mol = io.read('structure_2.xyz', parallel=False)
# mol = io.read('relax_final.xyz', parallel=False)
mol.calc = PySCF()
potentialenergy = mol.get_potential_energy()

vib = Vibrations(mol)
vib.run()
vib_energies = vib.get_energies()

thermo = IdealGasThermo(vib_energies=vib_energies,
                        potentialenergy=potentialenergy,
                        atoms=mol,
                        geometry='nonlinear',
                        symmetrynumber=3, spin=0)
G = thermo.get_gibbs_energy(temperature=298.15, pressure=101325.)

print("Vibrational energies: ", vib_energies)
