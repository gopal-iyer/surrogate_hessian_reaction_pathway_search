#! /usr/bin/env python

from pyscf import dft
from parameters import forward, backward
import numpy as np

### generated system text ###
from pyscf import gto as gto_loc
mol = gto_loc.Mole()

bond_length_grid = np.linspace(0.9, 1.1, 21)
bond_angle_grid = np.linspace(45*np.pi/180., 135*np.pi/180., 21)

def get_scf_result(bl, ba):
    pos_array = backward([bl, ba])
    mol.atom     = [['N', (pos_array[0][0], pos_array[0][1], pos_array[0][2])],
                    ['H', (pos_array[1][0], pos_array[1][1], pos_array[1][2])],
                    ['H', (pos_array[2][0], pos_array[2][1], pos_array[2][2])],
                    ['H', (pos_array[3][0], pos_array[3][1], pos_array[3][2])]]
    mol.basis    = 'ccecpccpvtz'
    mol.unit     = 'A'
    mol.ecp      = 'ccecp'
    mol.charge   = 0
    mol.spin     = 0
    mol.symmetry = False
    mol.build()
    
    mf = dft.RKS(mol)
    mf.xc          = 'pbe'
    mf.tol         = '1e-10'
    e_scf = mf.kernel()
    
    return e_scf

for i_bl, bl in enumerate(bond_length_grid):
    for i_ba, ba in enumerate(bond_angle_grid):
        e = get_scf_result(bl, ba)
        save_fname = f'p0_{bl:.3f}_p1_{ba:.3f}.dat'
        np.savetxt(save_fname, [e])
        