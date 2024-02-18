#!/usr/bin/env python3

from numpy import array, sin, cos, pi, mean, sign, dot
from nexus import obj, job, generate_physical_system, generate_pyscf, generate_convert4qmc, generate_qmcpack
from copy import deepcopy
from functools import partial
from machines import Workstation

cores_desired = [416, 256, 264]
for c in cores_desired:
    Workstation('ws'+str(c), c, 'mpirun')

Bohr = 0.5291772105638411

# nexus settings
nx_settings = obj(
    sleep         = 3,
    pseudo_dir    = 'pseudos/',
    runs          = '',
    results       = '',
    status_only   = 0,
    generate_only = 0,
    machine       = 'ws256',
)

from surrogate_macros import init_nexus
from surrogate_classes import mean_distances, distance, bond_angle
init_nexus(**nx_settings) # initiate nexus

# pos,cell given in angstrom
def forward(pos, axes = None):
    C, H0, H1, H2, F0, F1 = tuple(pos.reshape(-1, 3))
    CF0_bond_axis = deepcopy(C)
    CF0_bond_axis[0] = -1.
    CF0_bond_axis[1] =  0.
    CF0_bond_axis[2] =  0.
    r_CH = mean_distances([(C, H0),(C, H1),(C, H2)])
    r_CF0 = mean_distances([(C, F0)])
    r_CF1 = mean_distances([(C, F1)])
    a_HCF0 = mean([bond_angle(H0, C, CF0_bond_axis, units='rad'), bond_angle(H1, C, CF0_bond_axis, units='rad'),
                            bond_angle(H2, C, CF0_bond_axis, units='rad')]) #GRI IMPORTANT! changing angle units to rad
    return array([r_CH, a_HCF0, r_CF0, r_CF1])
#end def

def backward(params):
    r_CH, a_HCF0, r_CF0, r_CF1 = tuple(params)
    def backward_aux(p):
        x0, yz0, x1, x2 = tuple(p)
        C = array([0.0, 0.0, 0.0])
        H0 = array([ -x0, yz0, 0.0])
        H1 = array([ -x0, -yz0 * sin(pi / 6), yz0 * cos(pi / 6)])
        H2 = array([ -x0, -yz0 * sin(pi / 6), -yz0 * cos(pi / 6)])
        F0 = array([-x1, 0.0, 0.0])
        F1 = array([x2, 0.0, 0.0])
        pos = array([C, H0, H1, H2, F0, F1])
        return pos  # return parameters
    #end def
    def forward_aux(p):
        return forward(backward_aux(p))
    #end def
    theta = a_HCF0 # approximation
    #x0 = r_NH * cos(theta * pi / 180)
    #yz0 = r_NH * sin(theta * pi / 180)
    x0 = r_CH * cos(theta) #GRI switching to radians for meaningful non-diagonality
    yz0 = r_CH * sin(theta) #GRI switching to radians for meaningful non-diagonality
    x1 = r_CF0
    x2 = r_CF1
    p0 = [x0, yz0, x1, x2]
    from surrogate_classes import invert_pos
    p = invert_pos(pos0 = p0, params = params, forward = forward_aux, tol = 1e-9)
    return backward_aux(p)
#end def

def forward_subspace(curr_str, subspace_vectors_in_parameter_space, pos):
    forward_params = forward(pos)
    disp_vec = forward_params - curr_str
    forward_components = []
    for i_sv, sv in enumerate(subspace_vectors_in_parameter_space):
        forward_components.append(dot(disp_vec, sv))
    return array(forward_components)
#end def

def backward_subspace(curr_str, subspace_vectors_in_parameter_space, params):
    pos = deepcopy(curr_str)
    for i_p, p in enumerate(params):
        pos += p * subspace_vectors_in_parameter_space[i_p]
    backward_cartesian = backward(pos)
    return backward_cartesian
#end def

def get_pes_job(
    structure,
    path,
    template = 'pes_template.py',
    **kwargs
):
    system = generate_physical_system(
        structure = structure,
        net_charge = -1,
        C = 4,
        H = 1,
        F = 7,
        # Cl = 7
    )
    scf = generate_pyscf(
        template   = template,
        identifier = 'scf',
        path       = path,
        job        = job(serial=True,app='python3',hours=1),
        system     = system,
        mole       = obj(
            #GRI I guess this is all the change needed: Set ecp='ccecp', and don't specify basis at all
            #basis = 'bfd-vtz', #GRI change
            #basis = 'ccecp',
            ecp = 'ccecp', #GRI change
            basis = 'ccecpccpvtz',
            symmetry = False,
        ),
    )
    return [scf]
#end def

def get_pes_springs_job(
    structure,
    path,
    template = 'pes_springs_template.py',
    **kwargs
):
    print(structure)
    system = generate_physical_system(
        structure = structure,
        net_charge = -1,
        C = 4,
        H = 1,
        F = 7,
        # Cl = 7
    )
    scf = generate_pyscf(
        template   = template,
        identifier = 'scf',
        path       = path,
        job        = job(serial=True,app='python3',hours=1),
        system     = system,
        mole       = obj(
            #GRI I guess this is all the change needed: Set ecp='ccecp', and don't specify basis at all
            #basis = 'bfd-vtz', #GRI change
            #basis = 'ccecp',
            ecp = 'ccecp', #GRI change
            basis = 'ccecpccpvtz',
            symmetry = False,
        ),
    )
    return [scf]
#end def

cores = 256
#qmcapp = 'qmcpack'
qmcapp = 'qmcpack_complex'
convapp = 'convert4qmc'
qmcjob = obj(app=qmcapp, cores=cores, ppn=cores, hours=96)
convjob = obj(app=convapp, cores=1, hours=1)

from surrogate_macros import dmc_steps
def dmc_pes_job(
    dmc_enhancement_factor,
    structure,
    path,
    sigma = 0.01,
    var_eff = 1.0,
    template = 'pes_template.py',
    qmcpseudos = ['C.ccECP.xml', 'H.ccECP.xml', 'F.ccECP.xml'],
    **kwargs
):
    system = generate_physical_system(
        structure = structure,
        net_charge = -1,
        C = 4,
        H = 1,
        F = 7,
        # Cl = 7
    )
    scf = generate_pyscf(
        template   = template,
        identifier = 'scf',
        path       = path + '/scf',
        job        = job(serial=True,app='python3',hours=1),
        system     = system,
        mole       = obj(
            #basis = 'bfd-vtz', # JT: change to ccECP! #GRI change
            #ecp = 'bfd', # JT: change to ccECP! #GRI change
            ecp = 'ccecp',
            basis = 'ccecpccpvtz', #GRI!! This is needed. Only commenting it out for a test
            symmetry = False,
        ),
        save_qmc   = True,
    )
    c4q = generate_convert4qmc(
        identifier   = 'c4q',
        path         = path + '/scf', #GRI had to do conversion and SCF in the same folder because opt was not detecting the scf.h5 file for some reason
        #job          = job(cores=1,hours=1),
        job          = job(**convjob),
        #job          = job(serial=True, app=convapp, minutes=5),
        add_cusp     = False,
        dependencies = (scf,'orbitals'),
    )
    opt = generate_qmcpack(
        system       = system,
        path         = path+'/opt',
        job          = job(**qmcjob),
        dependencies = [(c4q,'orbitals')], #GRI had to do conversion and SCF in the same folder because opt was not detecting the scf.h5 file for some reason
        cycles       = 8,
        identifier   = 'opt',
        qmc          = 'opt',
        input_type   = 'basic',
        pseudos      = qmcpseudos,
        bconds       = 'nnn',
        J2           = True,
        J1_size      = 6,
        J1_rcut      = 6.0,
        J2_size      = 8,
        J2_rcut      = 8.0,
        minmethod    = 'oneshift',
        blocks       = 200,
        substeps     = 2,
        steps        = 1,
        samples      = 100000,
        minwalkers   = 0.1,
        nonlocalpp   = True,
        use_nonlocalpp_deriv = True,
    )
    dmcsteps = dmc_steps(sigma, var_eff = var_eff)
    # print(var_eff, dmcsteps, sigma)
    dmc = generate_qmcpack(
        system       = system,
        path         = path+'/dmc',
        job          = job(**qmcjob),
        dependencies = [(c4q,'orbitals'),(opt,'jastrow') ],
        steps        = dmc_enhancement_factor * dmcsteps,
        identifier   = 'dmc',
        qmc          = 'dmc',
        input_type   = 'basic',
        pseudos      = qmcpseudos,
        bconds       = 'nnn',
        jastrows     = [],
        vmc_samples  = 1000,
        blocks       = 200,
        timestep     = 0.005,
        nonlocalmoves= True,
        ntimesteps   = 1,
    )
    return [scf,c4q,opt,dmc]
#end def

