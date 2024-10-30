import numpy as np
import torch


def batch_triangulate_dlt_ransac_torch(kp2ds, Ks, Extrs, n_cams=3, n_iterations=100, reprojection_threshold=10.0):
    """
    Triangulate multiple 2D points from multiple sets of multiviews using the DLT algorithm with RANSAC.
    
    Args:
        kp2ds (torch.Tensor): Shape: (B, N, J, 2).
        Ks (torch.Tensor): Shape: (B, N, 3, 3).
        Extrs (torch.Tensor): Shape: (B, N, 4, 4).
        n_iterations (int): Number of RANSAC iterations.
        reprojection_threshold (float): Threshold for considering a point as an inlier.
    
    Returns:
        torch.Tensor: Shape: (B, J, 3), triangulated points.
    """
    import random
    import itertools

    nJoints = kp2ds.shape[-2]
    batch_size = kp2ds.shape[0]
    total_cams = kp2ds.shape[1]
    combinations = list(itertools.combinations(range(total_cams), n_cams))
    random.shuffle(combinations)  # randomize
    iters = min(len(combinations), n_iterations)

    # Pre-compute projection matrices for all cameras
    Pmat = Extrs[..., :3, :]  # (B, N, 3, 4)
    Mmat = torch.matmul(Ks, Pmat)  # (B, N, 3, 4)
    
    best_X = torch.zeros(batch_size, nJoints, 3, device=kp2ds.device)
    best_inlier_count = torch.zeros(batch_size, nJoints, device=kp2ds.device, dtype=torch.long)

    for idx in range(iters):
        # Randomly select n_cams cameras for triangulation
        # cam_indices = torch.randperm(total_cams)[:n_cams]
        cam_indices = combinations[idx]
        # print(cam_indices)
        
        # Extract keypoints and projection matrices for selected cameras
        selected_kp2ds = kp2ds[:, cam_indices, ...]  # (B, 2, J, 2)
        selected_Mmat = Mmat[:, cam_indices, ...]  # (B, 2, 3, 4)

        # Perform triangulation with the selected cameras
        X = batch_triangulate_dlt_torch(selected_kp2ds, Ks[:, cam_indices, ...], Extrs[:, cam_indices, ...])
        
        # Calculate reprojection error for all cameras
        reprojection_errors, inlier_count = calculate_reprojection_error(X, kp2ds, Mmat, reprojection_threshold)
        # print(f"iter: {idx}, inliers: {inlier_count[0, 0]}, repr error: {reprojection_errors[0, :, 0]}")
        
        # Update best triangulation if current iteration has more inliers
        update_mask = inlier_count > best_inlier_count
        best_X[update_mask] = X[update_mask]
        best_inlier_count[update_mask] = inlier_count[update_mask]

    return best_X


def calculate_reprojection_error(X, kp2ds, Mmat, reprojection_threshold=1.0):
    """
    Calculate the reprojection error and count inliers for triangulated 3D points.

    Args:
        X (torch.Tensor): Triangulated 3D points, shape (B, J, 3).
        kp2ds (torch.Tensor): Original 2D keypoints, shape (B, N, J, 2).
        Mmat (torch.Tensor): Camera projection matrices, shape (B, N, 3, 4).
        reprojection_threshold (float): Threshold to consider a reprojection as an inlier.

    Returns:
        torch.Tensor: Reprojection errors for each point across all cameras.
        torch.Tensor: Count of inliers for each triangulated point.
    """
    B, J, _ = X.shape
    N = kp2ds.shape[1]

    # Homogenize X for multiplication: (B, J, 4)
    X_hom = torch.cat([X, torch.ones(B, J, 1, device=X.device)], dim=-1)

    # Project 3D points back to 2D for each camera: (B, N, J, 3)
    X_proj = torch.einsum('bnik,bjk->bnji', Mmat, X_hom)

    # Convert homogeneous 2D points to cartesian: (B, N, J, 2)
    X_proj_cart = X_proj[..., :2] / X_proj[..., 2:3]

    # Compute reprojection error: (B, N, J)
    reprojection_errors = torch.norm(kp2ds - X_proj_cart, dim=-1)

    # Count inliers: (B, J)
    inliers = reprojection_errors < reprojection_threshold
    # print(inliers.shape, "inliners")
    inlier_counts = inliers.sum(dim=1)

    return reprojection_errors, inlier_counts


# source: https://github.com/lixiny/POEM/blob/main/lib/utils/triangulation.py
def batch_triangulate_dlt_torch(kp2ds, Ks, Extrs):
    """torch: Triangulate multiple 2D points from multiple sets of multiviews using the DLT algorithm.
    NOTE: Expand to Batch and nJoints dimension.

    see Hartley & Zisserman section 12.2 (p.312) for info on SVD,
    see Hartley & Zisserman (2003) p. 593 (see also p. 587).

    Args:
        kp2ds (torch.Tensor): Shape: (B, N, J, 2).
        Ks (torch.Tensor): Shape: (B, N, 3, 3).
        Extrs (torch.Tensor): Shape: (B, N, 4, 4). (world-to-camera)

    Returns:
        torch.Tensor: Shape: (B, J, 3).
    """
    # assert kp2ds.shape[0] == Ks.shape[0] == Extrs.shape[0], "batch shape mismatch"
    # assert kp2ds.shape[1] == Ks.shape[1] == Extrs.shape[1], "nCams shape mismatch"
    # assert kp2ds.shape[-1] == 2, "keypoints must be 2D"
    # assert Ks.shape[-2:] == (3, 3), "K must be 3x3"
    # assert Extrs.shape[-2:] == (4, 4), "Extr must be 4x4"

    nJoints = kp2ds.shape[-2]
    batch_size = kp2ds.shape[0]
    nCams = kp2ds.shape[1]

    Pmat = Extrs[..., :3, :]  # (B, N, 3, 4)
    Mmat = torch.matmul(Ks, Pmat)  # (B, N, 3, 4)
    Mmat = Mmat.unsqueeze(1).repeat(1, nJoints, 1, 1, 1)  # (B, J, N, 3, 4)
    Mmat = Mmat.reshape(batch_size * nJoints, nCams, *Pmat.shape[-2:])  # (BxJ, N, 3, 4)
    M_row2 = Mmat[..., 2:3, :]  # (BxJ, N, 1, 4)

    # kp2ds: (B, N, J, 2) -> (B, J, N, 2) -> (BxJ, N, 2) -> (BxJ, N, 2, 1)
    kp2ds = kp2ds.permute(0, 2, 1, 3).reshape(batch_size * nJoints, nCams, 2).unsqueeze(3)  # (BxJ, N, 2, 1)
    A = kp2ds * M_row2  # (BxJ, N, 2, 4)
    A = A - Mmat[..., :2, :]  # (BxJ, N, 2, 4)
    A = A.reshape(batch_size * nJoints, -1, 4)  # (BxJ, 2xN, 4)

    U, D, VT = torch.linalg.svd(A)  # VT: (BxJ, 4, 4)
    X = VT[:, -1, :3] / (VT[:, -1, 3:] + 1e-7)  # (BxJ, 3) # normalize
    X = X.reshape(batch_size, nJoints, 3)  # (B, J, 3)
    return X


def triangulate_dlt_torch(kp2ds, Ks, Extrs):
    """torch: Triangulate multiple 2D points from one set of multiviews using the DLT algorithm.
    NOTE: Expand to nJoints dimension.

    see Hartley & Zisserman section 12.2 (p.312) for info on SVD,
    see Hartley & Zisserman (2003) p. 593 (see also p. 587).

    Args:
        kp2ds (torch.Tensor): Shape: (N, J, 2).
        Ks (torch.Tensor): Shape: (N, 3, 3).
        Extrs (torch.Tensor): Shape: (N, 4, 4). (world-to-camera)

    Returns:
        torch.Tensor: Shape: (J, 3).
    """
    nJoints = kp2ds.shape[-2]

    Pmat = Extrs[:, :3, :]  # (N, 3, 4)
    Mmat = torch.matmul(Ks, Pmat)  # (N, 3, 4)
    Mmat = Mmat.unsqueeze(0).repeat(nJoints, 1, 1, 1)  # (J, N, 3, 4)
    M_row2 = Mmat[..., 2:3, :]  # (J, N, 1, 4)

    kp2ds = kp2ds.permute(1, 0, 2).unsqueeze(3)  # (J, N, 2, 1)
    A = kp2ds * M_row2  # (J, N, 2, 4)
    A = A - Mmat[..., :2, :]  # (J, N, 2, 4)
    A = A.reshape(nJoints, -1, 4)  # (J, 2xN, 4)

    U, D, VT = torch.linalg.svd(A)  # VT: (J, 4, 4)
    X = VT[:, -1, :3] / VT[:, -1, 3:]  # (J, 3) # normalize
    return X


def triangulate_one_point_dlt(points_2d_set, Ks, Extrs):
    """Triangulate one point from one set of multiviews using the DLT algorithm.
    Implements a linear triangulation method to find a 3D
    point. For example, see Hartley & Zisserman section 12.2 (p.312).
    for info on SVD, see Hartley & Zisserman (2003) p. 593 (see also p. 587)

    Args:
        points_2d_set (set): first element is the camera index, second element is the 2d point, shape: (2,)
        Ks (np.ndarray): Camera intrinsics. Shape: (N, 3, 3).
        Extrs (np.ndarray): Camera extrinsics. Shape: (N, 4, 4).

    Returns:
        np.ndarray: Triangulated 3D point. Shape: (3,).
    """
    A = []
    for n, pt2d in points_2d_set:
        K = Ks[int(n)]  # (3, 3)
        Extr = Extrs[int(n)]  # (4, 4)
        P = Extr[:3, :]  # (3, 4)
        M = K @ P  # (3, 4)
        row_2 = M[2, :]
        x, y = pt2d[0], pt2d[1]
        A.append(x * row_2 - M[0, :])
        A.append(y * row_2 - M[1, :])
    # Calculate best point
    A = np.array(A)
    u, d, vt = np.linalg.svd(A)
    X = vt[-1, 0:3] / vt[-1, 3]  # normalize
    return X


def triangulate_dlt(pts, confis, Ks, Extrs, confi_thres=0.5):
    """Triangulate multiple 2D points from one set of multiviews using the DLT algorithm.
    Args:
        pts (np.ndarray): 2D points in the image plane. Shape: (N, J, 2).
        confis (np.ndarray): Confidence scores of the points. Shape: (N, J,).
        Ks (np.ndarray): Camera intrinsics. Shape: (N, 3, 3).
        Extrs (np.ndarray): Camera extrinsics. Shape: (N, 4, 4).
        confi_thres (float): Threshold of confidence score.
    Returns:
        np.ndarray: Triangulated 3D points. Shape: (N, J, 3).
    """

    assert pts.ndim == 3 and pts.shape[-1] == 2
    assert confis.ndim == 2 and confis.shape[0] == pts.shape[0]
    assert Ks.ndim == 3 and Ks.shape[1:] == (3, 3)
    assert Extrs.ndim == 3 and Extrs.shape[1:] == (4, 4)
    assert Ks.shape[0] == Extrs.shape[0] == pts.shape[0]

    dtype = pts.dtype
    nJoints = pts.shape[1]
    p3D = np.zeros((nJoints, 3), dtype=dtype)

    for j, conf in enumerate(confis.T):
        while True:
            sel_cam_idx = np.where(conf > confi_thres)[0]
            if confi_thres <= 0:
                break
            if len(sel_cam_idx) <= 1:
                confi_thres -= 0.05
                # print('confi threshold too high, decrease to', confi_thres)
            else:
                break
        points_2d_set = []
        for n in sel_cam_idx:
            points_2d = pts[n, j, :]
            points_2d_set.append((str(n), points_2d))
        p3D[j, :] = triangulate_one_point_dlt(points_2d_set, Ks, Extrs)
    return p3D
