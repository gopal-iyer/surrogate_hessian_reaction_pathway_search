#! /usr/bin/env python

from pyscf import dft

$system

mf = dft.RKS(mol)
mf.xc          = 'pbe'
mf.tol         = '1e-10'
e_scf = mf.kernel()

from numpy import savetxt
savetxt('energy.dat', [e_scf, 0.0])
