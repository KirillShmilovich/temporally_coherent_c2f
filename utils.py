import math
import numpy as np
import torch


def _facify(n, fac):
    return int(n // fac)


def compute_same_padding(kernel_size, stride, dilation):
    if kernel_size % 2 == 0:
        raise ValueError("Only wörks for odd kernel sizes.")
    out = math.ceil((1 - stride + dilation * (kernel_size - 1)) / 2)
    return max(out, 0)


def voxelize_gauss(coord_inp, sigma, grid):
    coords = coord_inp[..., None, None, None]
    voxels = np.exp(-1.0 * np.sum((grid - coords) * (grid - coords), axis=2) / sigma)
    # voxels = np.transpose(voxels, [0, 2, 3, 4, 1])
    return voxels.squeeze()


def make_grid_np(ds, grid_size):
    grid = np.arange(-int(grid_size / 2), int(grid_size / 2), 1.0).astype(np.float32)
    grid += 0.5
    grid *= ds
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing="ij")
    grid = np.stack([X, Y, Z])[None, None, ...]
    return grid


def rand_rotation_matrix(deflection=1.0, randnums=None):
    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def avg_blob(grid, res, width, sigma, device=None):
    X, Y, Z = make_grid(res, width, device=device)
    X = X.view(1, 1, *X.shape)
    Y = Y.view(1, 1, *Y.shape)
    Z = Z.view(1, 1, *Z.shape)
    reduction_dims = (-1, -2, -3)
    grid = grid / torch.sum(grid, dim=reduction_dims, keepdim=True)
    X = torch.sum(grid * X, dim=reduction_dims)
    Y = torch.sum(grid * Y, dim=reduction_dims)
    Z = torch.sum(grid * Z, dim=reduction_dims)
    coords = torch.stack((X, Y, Z), dim=2)
    return coords


def make_grid(res, width, device=None):
    grid = (width / res) * (
        torch.arange(-int(res / 2), int(res / 2), device=device, dtype=torch.float)
        + 0.5
    )
    return torch.meshgrid(grid, grid, grid)


def voxel_gauss(coords, res, width, sigma, device=None):
    grid = torch.stack(make_grid(res=res, width=width, device=device)).unsqueeze(0)
    coords = coords.view(*coords.shape, 1, 1, 1)
    grid = torch.exp(-torch.div(torch.sum((grid - coords) ** 2, dim=2), sigma))
    return grid


def to_difference_matrix(X):
    return X.unsqueeze(2) - X.unsqueeze(1)


def to_distmat(x):
    diffmat = to_difference_matrix(x)
    return torch.sqrt(
        torch.clamp(torch.sum(torch.square(diffmat), axis=-1, keepdims=False), min=1e-8)
    )


def rigid_transform(A, B):
    """Returns the rotation matrix (R) and translation (t) to solve 'A @ R + t = B'
    A,B are N x 3 matricies"""
    # http://nghiaho.com/?page_id=671

    centoid_A = A.mean(0, keepdims=True)
    centoid_B = B.mean(0, keepdims=True)

    H = (A - centoid_A).T @ (B - centoid_B)

    U, S, Vt = np.linalg.svd(H, full_matrices=False)

    R = Vt.T @ U.T  # 3x3

    # Reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    R = R.T  # Transpose because A is not 3 x N

    t = centoid_B - centoid_A @ R  # 1x3

    return R, t
