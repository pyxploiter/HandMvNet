import numpy as np


def rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])
    return np.dot(points, ry)


def rigid_transform_3D(A, B):
    '''
    Returns a rigid 3D transformation in terms of 3D rotation and translation needed to bring
    point-set A to point-set B, both defined as a numpy array of 3 component vectors shape=(3,n).

    Returns
    -------
    R: ndarray
        3x3 rotation matrix
    t: ndarray
        3x1 translation vector
    '''
    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def param_count(net):
    return sum(p.numel() for p in net.parameters()) / 1e6


def param_size(net):
    # ! treat all parameters to be float
    return sum(p.numel() for p in net.parameters()) * 4 / (1024 * 1024)