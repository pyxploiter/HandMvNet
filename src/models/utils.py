import torch
import numpy as np
import scipy.sparse as sp
from torch.nn import functional as F


def soft_argmax_3d(heatmap3d, temperature: float=1000):
    batch_size = heatmap3d.shape[0]
    depth, height, width = heatmap3d.shape[2:]
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth*height*width))
    heatmap3d = F.softmax(heatmap3d*temperature, 2)
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))

    accu_x = heatmap3d.sum(dim=(2, 3))
    accu_y = heatmap3d.sum(dim=(2, 4))
    accu_z = heatmap3d.sum(dim=(3, 4))

    # Create coordinate tensors for x, y, z
    x_indices = torch.arange(width, dtype=torch.float32, device=heatmap3d.device)[None, None, :]
    y_indices = torch.arange(height, dtype=torch.float32, device=heatmap3d.device)[None, None, :]
    z_indices = torch.arange(depth, dtype=torch.float32, device=heatmap3d.device)[None, None, :]

    accu_x = accu_x * x_indices
    accu_y = accu_y * y_indices
    accu_z = accu_z * z_indices

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    return coord_out


def soft_argmax_2d(heatmap, temperature: float=1000):
    batch_size, num_joints, height, width = heatmap.shape

    # Reshape the 2D heatmap to (batch_size, num_joints, height * width)
    heatmap = heatmap.view(batch_size, num_joints, -1)

    # Apply softmax along the last dimension
    heatmap = F.softmax(heatmap*temperature, dim=2)

    # Reshape the heatmap back to its original shape
    heatmap = heatmap.view(batch_size, num_joints, height, width)

    # Calculate the coordinates along the x and y dimensions
    accu_x = heatmap.sum(dim=(2,))
    accu_y = heatmap.sum(dim=(3,))

    # Create coordinate tensors for x and y
    x_indices = torch.arange(width, dtype=torch.float32, device=heatmap.device)[None, None, :]
    y_indices = torch.arange(height, dtype=torch.float32, device=heatmap.device)[None, None, :]

    # Compute the expected x and y coordinates
    expected_x = (accu_x * x_indices).sum(dim=2, keepdim=True)
    expected_y = (accu_y * y_indices).sum(dim=2, keepdim=True)

    # Concatenate the expected x and y coordinates to get (batch_size, num_joints, 2)
    coordinates = torch.cat((expected_x, expected_y), dim=2)

    return coordinates


def heatmaps_to_coordinates(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_sparse_matrix(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize_sparse_matrix(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


def mask_joints(joints, mask):
    # mask: [b, v, 21]
    # joints: [b, v, 21, 2]
    # Expand the mask tensor to have the same shape as the joints tensor
    expanded_mask = mask.unsqueeze(-1).expand_as(joints)  # [b, v, 21, 2]
    # Element-wise multiplication to mask the joints
    masked_joints = joints * (~expanded_mask)

    return masked_joints


def generate_centered_coordinates(coordinates, px, py):
    """
    Calculates centered coordinates for a set of points.

    Args:
      coordinates: Tensor of shape (batch_size, num_points, 2) where the last 
                  dimension represents (x, y) coordinates.
      px: x-coordinate of the principal point.
      py: y-coordinate of the principal point.

    Returns:
      Tensor of shape (batch_size, num_points, 2) representing 
                      x-coordinates and y-coordinates centered at the principal
                      point
    """
    x_centered = coordinates[:, :, 0] - px.to(coordinates.device)
    y_centered = coordinates[:, :, 1] - py.to(coordinates.device)

    return torch.stack((x_centered, y_centered), dim=2)


def generate_fov_map(centered_coords, fx, fy):
    """
    Calculates the field of view (FoV) map for a set of centered coordinates.

    Args:
      centered_coords: Tensor of shape (batch_size, num_points, 2) representing centered 
                  x-y coordinates.
      fx: Focal length in the x-direction.
      fy: Focal length in the y-direction.

    Returns:
      Tensor of shape (batch_size, num_points, 2) representing 
                      the horizontal and vertical FoV
    """
    theta_x = torch.atan(centered_coords[:, :, 0] / fx.to(centered_coords.device))
    theta_y = torch.atan(centered_coords[:, :, 1] / fy.to(centered_coords.device))
    return torch.stack((theta_x, theta_y), dim=2)


def batch_compute_similarity_transform_torch(S1, S2, R_GT = None):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # Construct R.
    if R_GT is None:
        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0],1,1)
        Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))
        R = V.bmm(Z.bmm(U.permute(0,2,1)))
    else:
        R = R_GT

    # # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1
    
    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t
    return S1_hat, (scale, R, t)