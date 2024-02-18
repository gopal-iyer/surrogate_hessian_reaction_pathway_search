import numpy as np

def gram_schmidt(vectors):
    """
    Perform the Gram-Schmidt process on a list of vectors.
    """
    orthonormal_basis = []
    for v in vectors:
        for u in orthonormal_basis:
            v -= u.dot(v) * u
        if np.linalg.norm(v) != 0:
            orthonormal_basis.append(v / np.linalg.norm(v))
    return orthonormal_basis

def orthogonal_subspace_basis(vec):
    """
    Find an orthonormal basis for the subspace orthogonal to 'vec'.
    """
    # Normalize 'vec'
    vec = vec / np.linalg.norm(vec)

    # Get the dimension of the space
    dim = len(vec)

    # Create an initial basis for the whole space including 'vec'
    initial_basis = [vec] + [np.eye(dim)[:,i] for i in range(dim)]

    # Orthonormalize the entire basis
    orthonormalized = gram_schmidt(initial_basis)

    # Exclude the orthonormal vector corresponding to 'vec'
    # We use a threshold to determine near-orthogonality because of potential floating point inaccuracies
    threshold = 1e-10
    subspace_basis = np.array([v for v in orthonormalized if abs(np.dot(v, vec)) < threshold])

    return subspace_basis

# vec = np.array([1, 0, 0])
# basis = orthogonal_subspace_basis(vec)
#
# # Print the basis for verification
# for b in basis:
#     print(b)
