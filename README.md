# About
This repository includes scripts to run the surrogate Hessian subspace optimization and saddle-point search algorithms based on the manuscript '[Force-free identification of minimum-energy pathways and transition states for stochastic electronic structure theories](https://arxiv.org/abs/2402.13189)' by Iyer, Whelpley, Tiihonen, Kent, Krogel, and Rubenstein (2024). This repository may be updated as needed based on user feedback.

# Prerequisites
To reproduce the results in the manuscript, you will need to have access to the original [surrogate Hessian line-search](https://github.com/QMCPACK/surrogate_hessian_relax/tree/master) code (and all associated Python and QMCPACK-related packages) and add it to your `PYTHONPATH`.

# Usage
Inside the `NH3_inversion` and `SN2_reaction` directories, there are extensive README files that will guide you through the various calculations step by step.
The subspace optimization algorithm can be found in directories with the suffix `_transition_pathway` and is mainly contained within `transition_pathway_automated.py` with some helper functions in `orthonormal_subspace.py` and `parameters.py`.
The subspace optimization algorithm works as follows:
1. Start with structures along a guess minimum-energy pathway (MEP) – here, this is obtained from a prior DFT-NEB calculation.
2. Parameterize these structures according to some structural parameterization scheme.
3. Compute tangents at each structure along the pathway and construct basis vectors for subspaces orthogonal to these tangents for each structure.
4. Construct a map from structural parameter space to the path-orthogonal subspace.
5. Compute the Hessian and perform the usual surrogate Hessian line-search within the subspace to identify the optimal structure on the path for each guess structure.

Once you are familiar with the algorithm, following the scripts should be straightforward.
