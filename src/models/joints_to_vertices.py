import torch
import numpy as np
from manopth.manolayer import ManoLayer

from utils.misc import rigid_transform_3D
from utils.analytical_ik import adaptive_IK


class JointsToVertices:
    def __init__(self, mano_dir, device="cpu"):
        self.device = device
        # TODO: manopth/rotproj.py has rotmat.cuda() call so only works with cuda device for now
        # Only fixed the local copy of the file for now
        self.mano_decoder = ManoLayer(flat_hand_mean=True,
                                      side="right",
                                      mano_root=mano_dir,
                                      use_pca=False,
                                      root_rot_mode="rotmat",
                                      joint_rot_mode="rotmat",
                                      )
        self.mano_decoder.to(device)
        # flat hand template
        self.joints_template = self.mano_decoder(torch.eye(3).repeat(1, 16, 1, 1).to(device))[1].squeeze().cpu().numpy()

    def __call__(self, joints):
        # estimate bones directions and find corresponding interpolated parameters
        computed_pts = np.zeros((3, 3))
        mano_pts = np.zeros((3, 3))

        computed_pts[:, 0] = joints[0]
        computed_pts[:, 1] = joints[9]
        computed_pts[:, 2] = joints[13]

        mano_pts[:, 0] = self.joints_template[0]
        mano_pts[:, 1] = self.joints_template[9]
        mano_pts[:, 2] = self.joints_template[13]

        ret_R, ret_t = rigid_transform_3D(computed_pts, mano_pts)
        joints_to_mano = ((ret_R@joints.T) + ret_t).T

        pose_R = adaptive_IK(self.joints_template, joints_to_mano)
        pose_R = torch.from_numpy(pose_R).float().to(self.device)

        hand_verts, _ = self.mano_decoder(pose_R)
        mano_v = hand_verts.cpu().numpy()[0]

        # re-position mano vertices to reach the original joints position
        mano_v = (np.linalg.inv(ret_R)@(mano_v.T - ret_t)).T

        return mano_v
