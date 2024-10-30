import torch


def transform_joints_between_cameras(joint_locations, source_extrinsic, target_extrinsic):
    """
    Transform joint locations from one camera's coordinate system to another.
    :param joint_locations: PyTorch tensor of joint locations in the source camera's coordinate system.
    :param source_extrinsic: PyTorch tensor (4,4) representing the source camera's extrinsic parameters.
    :param target_extrinsic: PyTorch tensor (4,4) representing the target camera's extrinsic parameters.
    :return: Transformed joint locations in the target camera's coordinate system.
    """
    # Convert to homogeneous coordinates
    ones = torch.ones((joint_locations.shape[0], 1), device=joint_locations.device)
    homogeneous_joint_locations = torch.cat((joint_locations, ones), dim=1)

    # Transform to world coordinates (using inverse of source extrinsics)
    world_coordinates = torch.matmul(source_extrinsic, homogeneous_joint_locations.T).T

    # Transform from world to target camera coordinates
    target_coordinates = torch.matmul(torch.inverse(target_extrinsic), world_coordinates.T).T
    # Drop the homogeneous coordinate
    return target_coordinates[:, :3]


def get_2d_joints_from_3d_joints(joints_3d, root_idx, intrinsics, extrinsics):
    """
    Get 2D joint locations in pixels for each camera view.
    :param joints_3d: Torch tensor of shape (batch, 21, 3), Absolute 3D joints in the first camera view [unit=meters].
    :param root_idx: Index of the camera whose 3D joints are given.
    :param intrinsics: Torch tensor of shape (batch, num_views, 4), camera intrinsics [fx, fy, cx, cy].
    :param extrinsics: Torch tensor of shape (batch, num_views, 4, 4), camera extrinsics.
    :return: 2D joints in pixels for each camera view, shape (batch, num_views, 21, 2).
    """
    batch_size, num_views, num_joints = joints_3d.shape[0], intrinsics.shape[1], joints_3d.shape[1]
    joints_2d = torch.zeros(batch_size, num_views, num_joints, 2, device=joints_3d.device)

    for i in range(num_views):
        for b in range(batch_size):
            # Transform joints to current camera's coordinate system
            transformed_joints = transform_joints_between_cameras(joints_3d[b], extrinsics[b, root_idx], extrinsics[b, i])
            # Project to 2D
            joints_2d[b, i] = camera_to_image_projection(transformed_joints*1000, intrinsics[b, i])[:, :2]

    return joints_2d


def camera_to_image_projection(points, camera, scale=(1, 1), epsilon=1e-6):
    # for all points
    # ndarray (n, 3) - xyz
    # camera = [fx, fy, cx, cy]
    # returns uvd in image coords

    # Add epsilon to the Z coordinate to avoid division by zero
    z_modified = points[:, 2] + epsilon

    x = (points[:, 0] * camera[0] / z_modified + camera[2]) * scale[0]
    y = (points[:, 1] * camera[1] / z_modified + camera[3]) * scale[1]

    points_2d = torch.stack((x, y, points[:, 2]), dim=1)  # No in-place operation
    return points_2d


def image_to_camera_projection(points, camera, scale=(1, 1)):
    # for all points
    # ndarray (n,3) - uvd
    # camera = [fx, fy, cx, cy]
    # returns xyz in camera coords
    
    x = (points[:, 0] * scale[0] - camera[2]) * points[:, 2] / camera[0]
    y = (points[:, 1] * scale[1] - camera[3]) * points[:, 2] / camera[1]
    points_3d = torch.stack((x, y, points[:, 2]), dim=1)  # No in-place operation
    return points_3d


def cam_to_world(X_cam, T_cam2world):
    # Add homogeneous coordinate to camera points
    X_cam_hom = torch.cat([X_cam, torch.ones((X_cam.shape[0], 1), device=X_cam.device)], dim=-1)
    # Transform to world coordinates
    X_world = (T_cam2world @ X_cam_hom.T).T
    # Return only the x, y, z components
    return X_world[:, :3]


def world_to_cam(X_world, T_cam2world):
    # Calculate the inverse of the extrinsic matrix
    T_world2cam = torch.linalg.inv(T_cam2world)
    # Add homogeneous coordinate to world points
    X_world_hom = torch.cat([X_world, torch.ones((X_world.shape[0], 1), device=X_world.device)], dim=-1)
    # Transform to camera coordinates
    X_cam = (T_world2cam @ X_world_hom.T).T
    # Return only the x, y, z components
    return X_cam[:, :3]


def cam1_to_cam2(X_cam1, T_cam1_to_world, T_cam2_to_world):
    # Convert point from camera 1 to world coordinates
    X_cam1_hom = torch.cat([X_cam1, torch.ones((X_cam1.shape[0], 1), device=X_cam1.device)], dim=-1)
    X_world = (T_cam1_to_world @ X_cam1_hom.T).T
    
    # Inverse of camera 2 extrinsic matrix to go from world to camera 2
    T_world_to_cam2 = torch.linalg.inv(T_cam2_to_world)
    X_cam2 = (T_world_to_cam2 @ X_world.T).T
    
    # Return only the x, y, z components
    return X_cam2[:, :3]


def create_intrinsics_matrix(intrinsics):
    """
    Converts intrinsics tensor (num_views, 4) to K matrix (num_views, 3, 3).
    
    :param intrinsics: torch.Tensor of shape (num_views, 4) where each row is [fx, fy, cx, cy]
    :return: torch.Tensor of shape (num_views, 3, 3) representing the K matrix for each view
    """
    num_views = intrinsics.shape[0]
    K = torch.zeros((num_views, 3, 3), device=intrinsics.device)

    # Set fx, fy, cx, cy for each view in the K matrix
    K[:, 0, 0] = intrinsics[:, 0]  # fx
    K[:, 1, 1] = intrinsics[:, 1]  # fy
    K[:, 0, 2] = intrinsics[:, 2]  # cx
    K[:, 1, 2] = intrinsics[:, 3]  # cy
    K[:, 2, 2] = 1.0  # bottom-right element is always 1

    return K
