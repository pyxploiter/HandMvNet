from manopth.manolayer import ManoLayer


class ManoDecoder:
    def __init__(self, models_dir, device='cpu', eval_mode=True):
        self.rh_model = ManoLayer(mano_root=models_dir, ncomps=45, flat_hand_mean=False, use_pca=True)
        self.lh_model = ManoLayer(mano_root=models_dir, ncomps=45, side='left', flat_hand_mean=False, use_pca=True)

        self.lh_model.to(device)
        self.rh_model.to(device)
        if eval_mode:
            self.lh_model.eval()
            self.rh_model.eval()

    def decode(self, pose, shape, side):
        # pose: [b, 51] or [b, 48]
        # shape: [b, 10] or None
        trans = None
        if pose.shape[1] == 51:
            trans = pose[:, 48:51]

        if side == 'right':
            verts, joints = self.rh_model(th_pose_coeffs=pose[:, :48], th_betas=shape, th_trans=trans)
        elif side == 'left':
            verts, joints = self.lh_model(th_pose_coeffs=pose[:, :48], th_betas=shape, th_trans=trans)
        else:
            raise ValueError('side must be either "right" or "left"')

        # in milimeters
        return verts, joints  # torch.Size([b, 778, 3]) torch.Size([b, 21, 3])


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt

    # Generate shape parameters
    shape = torch.zeros(1, 10)
    # Generate pose parameters, including 3 values for global axis-angle rotation
    pose = torch.zeros(1, 48)

    mano_model = ManoDecoder(models_dir="src/mano")
    verts, joints = mano_model.decode(pose, shape, side="right")
    root_joint = joints[:, 0].clone().unsqueeze(1)
    joints -= root_joint
    verts -= root_joint
    verts = verts[0]
    joints = joints[0]
    print("root_joint:", joints[0])
    print("root_vert", verts[0])
    print("joints:", joints)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot joints (in red)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', label='Joints')

    # Plot vertices (in blue)
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='b', label='Vertices', alpha=0.6)

    # Add labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()