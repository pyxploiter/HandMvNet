import torch
import torch.nn as nn
from torch.nn import functional as F

from constants import HAND_EDGES
import models.utils as model_utils
from models.layers import make_conv_layers, make_linear_layers, ChebConv, GraphConv


__all__ = ["PoseNet", "SampleNet", "JointsDecoderGCN", "JointsDecoderNN"]


class PoseNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.conv = make_conv_layers(layers, kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, img_feat):
        joint_hm = self.conv(img_feat)  # [b*v, 21, 32, 32]
        joint_coord = model_utils.soft_argmax_2d(joint_hm)  # [b*v, 21, 2]
        return joint_coord, joint_hm


class SampleNet(nn.Module):
    def __init__(self, layers, reduce_after_sample=False):
        super().__init__()
        self.reduce_after_sample = reduce_after_sample
        if self.reduce_after_sample:
            self.conv = make_linear_layers(layers, relu_final=False)
        else:
            self.conv = make_conv_layers(layers, kernel=1, stride=1, padding=0)

    def _sample_joint_features_without_grid_sample(self, img_feat, joint_xy):
        # Step 1: Convert the 2D points to integer values
        points_rounded = joint_xy.round().long()
        x_indices = points_rounded[:, :, 0]
        y_indices = points_rounded[:, :, 1]

        # Step 2: Create a tensor of batch indices
        batch_indices = torch.arange(img_feat.size(0)).unsqueeze(1).repeat(1, x_indices.size(1))

        # Step 3: Use advanced indexing to sample the feature vectors
        sampled_features = img_feat[batch_indices, :, x_indices, y_indices]
        return sampled_features

    def _sample_joint_features(self, img_feat, joint_xy):
        height, width = img_feat.shape[2:]
        x = joint_xy[:, :, 0] / (width-1) * 2 - 1
        y = joint_xy[:, :, 1] / (height-1) * 2 - 1
        grid = torch.stack((x, y), 2)[:, :, None, :]
        img_feat = F.grid_sample(img_feat, grid, align_corners=True)[:, :, :, 0]  # batch_size, channel_dim, joint_num
        img_feat = img_feat.permute(0, 2, 1).contiguous()  # batch_size, joint_num, channel_dim
        return img_feat

    def forward(self, img_feats, joint_coords):
        if self.reduce_after_sample:
            img_feats_joints = self._sample_joint_features(img_feats, joint_coords)
            img_feats_joints = self.conv(img_feats_joints)
        else:
            img_feats = self.conv(img_feats)
            img_feats_joints = self._sample_joint_features(img_feats, joint_coords)
        # img_feats_joints = torch.cat([img_feats_joints, joint_coords], dim=2)
        return img_feats_joints


class GraphChebConvNet(nn.Module):
    def __init__(self, in_dim, out_dim=3, hidden_dim=128, num_layers=3, K=2):
        """

        Args:
            in_dim (int): Input feature dimension.
            out_dim (int): Output feature dimension (e.g., 3 for 3D coordinates).
            hidden_dim (int): Hidden layer dimensionality.
            num_layers (int): Number of GCN layers.
            K (int): Polynomial order (Chebyshev filter size).
        """
        super(GraphChebConvNet, self).__init__()

        self.num_layers = num_layers
        self.relu = nn.LeakyReLU()

        # dexycb/mano edges
        hand_edges = torch.tensor(HAND_EDGES, dtype=torch.long)

        self.joints_adj = model_utils.adj_mx_from_edges(num_pts=21, edges=hand_edges, sparse=False)

        # Dynamically create GCN layers
        gcn_layers = []
        # Input layer
        gcn_layers.append(ChebConv(in_dim, hidden_dim, K=K))

        # Hidden layers
        for _ in range(1, num_layers - 1):
            gcn_layers.append(ChebConv(hidden_dim, hidden_dim, K=K))

        # Output layer
        gcn_layers.append(ChebConv(hidden_dim, out_dim, K=K))

        self.gcn_layers = nn.ModuleList(gcn_layers)

    def forward(self, x):
        """
        Forward pass through the GCN layers.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_joints, in_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_joints, out_dim).
        """
        joints_feat = x
        for i, gcn in enumerate(self.gcn_layers):
            joints_feat = gcn(joints_feat, self.joints_adj)
            if i < self.num_layers - 1:  # Apply activation to all but the last layer
                joints_feat = self.relu(joints_feat)
        return joints_feat


class JointsDecoderGCN(nn.Module):
    def __init__(self, in_features, out_dim=3):
        super(JointsDecoderGCN, self).__init__()

        # dexycb/mano edges
        hand_edges = torch.tensor(HAND_EDGES, dtype=torch.long)

        self.joints_adj = model_utils.adj_mx_from_edges(num_pts=21, edges=hand_edges, sparse=False)

        self.relu = nn.LeakyReLU()
        self.joints_gcn1 = ChebConv(in_features, 256, K=2)
        self.joints_gcn2 = ChebConv(256, 64, K=2)
        self.joints_gcn3 = ChebConv(64, out_dim, K=2)

    def forward(self, x):

        joints_cam = self.relu(self.joints_gcn1(x, self.joints_adj))
        joints_cam = self.relu(self.joints_gcn2(joints_cam, self.joints_adj))
        joints_cam = self.joints_gcn3(joints_cam, self.joints_adj)

        return joints_cam


class JointsDecoderNN(nn.Module):
    def __init__(self, in_features, out_dim=3):
        super(JointsDecoderNN, self).__init__()

        self.joints_fc1 = nn.Linear(in_features, 64)
        self.relu = nn.LeakyReLU()
        self.joints_fc2 = nn.Linear(64, out_dim)

    def forward(self, x):
        joints_cam = self.relu(self.joints_fc1(x))
        joints_cam = self.joints_fc2(joints_cam)

        return joints_cam


class GraphConvNet(nn.Module):

    def __init__(self, in_features, out_features, nodes, activation=nn.ReLU()):
        super(GraphConvNet, self).__init__()

        self.A_hat = nn.Parameter(torch.Tensor(nodes, nodes), requires_grad=True)

        self.gconv1 = GraphConv(in_features, in_features//2)
        self.gconv2 = GraphConv(in_features//2, out_features, activation=activation)

    def forward(self, X):
        X_0 = self.gconv1(X, self.A_hat)
        X_1 = self.gconv2(X_0, self.A_hat)
        return X_1


class ResidualGraphConv(nn.Module):
    def __init__(self, input_dim, hid_dim, nodes, p_dropout=0.1, activation=nn.ReLU()):
        super(ResidualGraphConv, self).__init__()
        self.activation = activation

        self.gconv1 = GraphConvNet(input_dim, hid_dim, nodes)
        self.gconv2 = GraphConvNet(hid_dim, input_dim, nodes)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        if self.activation is not None:
            out = self.activation(out)
        out = self.gconv2(out)
        if self.activation is not None:
            return self.activation(residual + out)

        return residual + out