import cv2
import torch
import plotly
import numpy as np
from PIL import Image
from io import BytesIO
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from manopth.manolayer import ManoLayer

from utils.misc import rotateY
from vis.utils import reverse_transform
from utils.camera import camera_to_image_projection
from constants import HAND_EDGES, HAND_EDGES_COLORS


class HandPoseVisualizer:
    """
    A class used to visualize hand pose data.

    This class provides methods to visualize different aspects of hand pose data,
    including RGB images, depth images, mask images, heatmaps, 2D joint locations,
    and 3D joint locations and vertices.

    Attributes
    ----------
    batch : dict
        A dictionary containing the batch data. The shape of the data depends on the dataset.

    Methods
    -------
    visualize_rgb_image(sample_idx=0, show=False)
        Display the RGB image for the given batch index.

    visualize_full_rgb_image(sample_idx=0, show=False)
        Display the full RGB image for the given batch index.

    visualize_mask_image(sample_idx=0, show=False)
        Display the mask image for the given batch index.

    visualize_depth_image(sample_idx=0, show=False)
        Display the depth image for the given batch index.

    visualize_combined_heatmaps(sample_idx=0)
        Display combined heatmaps for all views.

    visualize_joints_2d_on_cropped_image(sample_idx=0, show=False)
        Display the rgb image with 2D joint locations and bones.

    visualize_joints_2d_on_full_image(sample_idx=0, show=False)
        Display the image full RGB with bboxes, 2D joint locations and bones.

    visualize_projected_joints_2d_on_full_image(sample_idx=0, show=False)
        Display the rgb image with 2D joint locations and bones.

    visualize_3d_joints(sample_idx=0)
        Display 3D joint locations.

    visualize_3d_vertices(sample_idx=0)
        Display 3D vertices.

    visualize_3d_joints_and_vertices(sample_idx=0)
        Display both 3D joint locations and vertices in the same plot.

    visualize_3d_absolute_joints_and_vertices(sample_idx=0)
        Display absolute 3D joint locations and vertices in the same plot.
    """
    
    hand_edges = torch.tensor(HAND_EDGES, dtype=torch.long)
    edge_colors = (torch.tensor(HAND_EDGES_COLORS) * 255).int()
    hand_faces = None
    
    def __init__(self, batch, mano_dir="src/mano"):
        """
        Initialize the visualizer with a batch from the dataset.
        Args:
            batch (dict): A dictionary containing the batch data. The shape of the data depends on the dataset.
        """
        self.batch = batch
        self.views = batch["selected_views"]
        self.intrinsic = batch["cam_params"]["intrinsic"]
        self.extrinsic = batch["cam_params"]["extrinsic"]

        self.rh_model = ManoLayer(mano_root=mano_dir, ncomps=45, flat_hand_mean=False, use_pca=True)
        HandPoseVisualizer.hand_faces = self.rh_model.th_faces

    def visualize_rgb_image(self, sample_idx=0, show=False):
        """
        Display the RGB image for the given batch index for all views.
        Args:
            sample_idx (int): Sample index in the batch. Default is 0.
            show (bool): Whether to show the image. Default is False.
        Returns:
            np.ndarray: Horizontal stack of RGB images.
        """
        rgb_images = self.batch["data"]["rgb"][sample_idx]
        return HandPoseVisualizer.plot_images(rgb_images,
                                              title="RGB Image",
                                              show=show)

    def visualize_full_rgb_image(self, sample_idx=0, show=False):
        """
        Display the RGB image for the given batch index for all views.
        Args:
            sample_idx (int): Sample index in the batch. Default is 0.
            show (bool): Whether to show the image. Default is False.
        Returns:
            np.ndarray: Horizontal stack of RGB images.
        """
        full_rgb_images = self.batch["data"]["full_rgb"][sample_idx]
        return HandPoseVisualizer.plot_images(full_rgb_images,
                                              title="Full RGB Image",
                                              show=show)

    def visualize_mask_image(self, sample_idx=0, show=False):
        """
        Display the mask image for the given batch index for all views.
        Args:
            sample_idx (int): Sample index in the batch. Default is 0.
            show (bool): Whether to show the image. Default is False.
        Returns:
            np.ndarray: Horizontal stack of mask images.
        """
        mask_images = self.batch["data"]["mask"][sample_idx]
        return HandPoseVisualizer.plot_images(mask_images,
                                              title="Mask Image",
                                              denormalize=False,
                                              scale=4,
                                              show=show)

    def visualize_depth_image(self, sample_idx=0, show=False):
        """
        Display the depth image for the given batch index for all views.
        Args:
            sample_idx (int): Sample index in the batch. Default is 0.
            show (bool): Whether to show the image. Default is False.
        Returns:
            np.ndarray: Horizontal stack of depth images.
        """
        depth_images = self.batch["data"]["depth"][sample_idx]
        return HandPoseVisualizer.plot_images(depth_images,
                                              title="Depth Image",
                                              denormalize=False,
                                              scale=4,
                                              show=show)

    def visualize_combined_heatmaps(self, sample_idx=0, scale=8, show=False):
        """
        Display combined heatmaps for all views.
        Args:
            sample_idx (int): Sample index in the batch. Default is 0.
        Returns:
            plt.figure: The figure containing the combined heatmaps.
        """
        all_views_heatmaps = self.batch["data"]["heatmap"][sample_idx]  # [v=2, j=21, h=32, w=32]
        combined_heatmaps = all_views_heatmaps.sum(dim=1)  # Summing across the joints for each view
        return HandPoseVisualizer.plot_heatmaps(combined_heatmaps,
                                                title="Combined Heatmaps",
                                                scale=scale,
                                                show=show)

    def visualize_joints_2d_on_cropped_image(self, sample_idx=0, show=False):
        """
        Display the rgb image with 2D joint locations and bones.
        Args:
            sample_idx (int): Sample index in the batch. Default is 0.
            show (bool): Whether to show the image. Default is False.
        Returns:
            np.ndarray: The RGB image with 2D joint locations and bones.
        """
        images = self.batch["data"]["rgb"][sample_idx]
        joints = self.batch["data"]["joints_crop_img"][sample_idx]
        title = "RGB with Cropped Joints and Bones"

        return HandPoseVisualizer.plot_images_with_joints(images,
                                                          joints,
                                                          title=title,
                                                          show=show)

    def visualize_joints_2d_on_full_image(self, sample_idx=0, show=False):
        """
        Display the image full RGB with bboxes, 2D joint locations and bones.
        Args:
            sample_idx (int): Sample index in the batch. Default is 0.
            show (bool): Whether to show the image. Default is False.
        Returns:
            np.ndarray: The full RGB image with bounding boxes, 2D joint locations and bones.
        """
        images = self.batch["data"]["full_rgb"][sample_idx]
        joints = self.batch["data"]["joints_img"][sample_idx]
        bboxes = self.batch["data"]["bboxes"][sample_idx]
        title = "Full RGB with Bboxes, Joints and Bones"

        return HandPoseVisualizer.plot_full_images_with_joints_and_bboxes(images,
                                                                          joints,
                                                                          bboxes,
                                                                          title=title,
                                                                          show=show)

    def visualize_projected_joints_2d_on_full_image(self, sample_idx=0, show=False):
        """
        Display the rgb image with 2D joint locations and bones.
        Args:
            sample_idx (int): Sample index in the batch. Default is 0.
            show (bool): Whether to show the image. Default is False.
        Returns:
            np.ndarray: The RGB image with 2D joint locations and bones.
        """
        images = self.batch["data"]["full_rgb"][sample_idx]
        bboxes = self.batch["data"]["bboxes"][sample_idx]
        joints = self.batch["data"]["all_joints_cam"][sample_idx] + self.batch["data"]["all_root_joints"][sample_idx]

        joints_2d = torch.zeros(len(self.views[sample_idx]), 21, 2)

        for i, _ in enumerate(joints):
            joints_2d[i] = camera_to_image_projection(joints[i].clone(), self.intrinsic[sample_idx][i])[:, :2]

        title = "Full RGB with Bboxes, Projected Joints and Bones"

        return HandPoseVisualizer.plot_full_images_with_joints_and_bboxes(images,
                                                                          joints_2d,
                                                                          bboxes,
                                                                          title=title,
                                                                          show=show)
    
    def visualize_projected_vertices_2d_on_full_image(self, sample_idx=0, show=False):
        """
        Display the rgb image with 2D vertices locations and bones.
        Args:
            sample_idx (int): Sample index in the batch. Default is 0.
            show (bool): Whether to show the image. Default is False.
        Returns:
            np.ndarray: The RGB image with 2D vertices locations and bones.
        """
        images = self.batch["data"]["full_rgb"][sample_idx]
        bboxes = self.batch["data"]["bboxes"][sample_idx]
        vertices = self.batch["data"]["all_vertices"][sample_idx] + self.batch["data"]["all_root_joints"][sample_idx]

        vertices_2d = torch.zeros(len(self.views[sample_idx]), 778, 2)

        for i, _ in enumerate(vertices):
            vertices_2d[i] = camera_to_image_projection(vertices[i].clone(), self.intrinsic[sample_idx][i])[:, :2]

        title = "Full RGB with Bboxes, Projected Joints and Bones"

        return HandPoseVisualizer.plot_full_images_with_vertices_and_bboxes(images,
                                                                          vertices_2d,
                                                                          bboxes,
                                                                          title=title,
                                                                          show=show)

    def visualize_3d_joints(self, sample_idx=0):
        """
        Display 3D joint locations.
        Args:
            sample_idx (int): Sample index in the batch. Default is 0.
        """
        joints = self.batch["data"]["joints_cam"][sample_idx]  # [j=21, d=3]
        HandPoseVisualizer.plot_3d_data(joints, title="Root-relative 3D Joints")

    def visualize_3d_vertices(self, sample_idx=0):
        """
        Display 3D vertices.
        Args:
            sample_idx (int): Sample index in the batch. Default is 0.
        """
        vertices = self.batch["data"]["vertices"][sample_idx]  # [verts=778, d=3]
        HandPoseVisualizer.plot_3d_data(vertices, title="Root-relative 3D Vertices", is_vertices=True)

    def visualize_3d_joints_and_vertices(self, sample_idx=0):
        """
        Display both 3D joint locations and vertices in the same plot.
        Args:
            sample_idx (int): Sample index in the batch. Default is 0.
        """
        joints = self.batch["data"]["joints_cam"][sample_idx]  # [j=21, d=3]
        vertices = self.batch["data"]["vertices"][sample_idx]  # [verts=778, d=3]
        HandPoseVisualizer.plot_3d_joints_and_vertices(joints, vertices, title="Root-relative 3D Joints and Vertices")

    def visualize_3d_absolute_joints_and_vertices(self, sample_idx=0):
        """
        Display absolute 3D joint locations and vertices in the same plot.
        Args:
            sample_idx (int): Sample index in the batch. Default is 0.
        """
        joints_relative = self.batch["data"]["joints_cam"][sample_idx]  # [j=21, d=3]
        vertices_relative = self.batch["data"]["vertices"][sample_idx]  # [verts=778, d=3]
        root_joint = self.batch["data"]["root_joint"][sample_idx]  # [3]

        # Convert to absolute positions
        joints_absolute = joints_relative + root_joint
        vertices_absolute = vertices_relative + root_joint

        # Call the plotting function with absolute positions
        HandPoseVisualizer.plot_3d_joints_and_vertices(joints_absolute, vertices_absolute, title="Absolute 3D Joints and Vertices")

    @staticmethod
    def _draw_joints_on_image(image, joints, point_size=6, edge_width=3):
        edge_colors = HandPoseVisualizer.edge_colors
        
        for j_idx, joint in enumerate(joints):
            cv2.circle(image, (int(joint[0]), int(joint[1])), point_size, tuple(edge_colors[j_idx].tolist()), -1)
        
        for e_idx, edge in enumerate(HandPoseVisualizer.hand_edges):
            # Draw lines between joints
            joint1 = joints[edge[0]]
            joint2 = joints[edge[1]]
            # print(image.dtype, image.shape, joint1, tuple(edge_colors[e_idx+1].tolist()))
            cv2.line(image,
                     (int(joint1[0]), int(joint1[1])), 
                     (int(joint2[0]), int(joint2[1])), 
                     tuple(edge_colors[e_idx+1].tolist()), # bone color
                     edge_width)
        
        return image
    
    @staticmethod
    def _draw_vertices_on_image(image, vertices2d):
        # Draw faces
        for face in HandPoseVisualizer.hand_faces:
            for fi in range(len(face)):
                start_vertex = vertices2d[face[fi]]
                end_vertex = vertices2d[face[(fi + 1) % len(face)]]
                cv2.line(image, (int(start_vertex[0]), int(start_vertex[1])), (int(end_vertex[0]), int(end_vertex[1])), color=(255, 255, 255), thickness=1)

        return image

    @staticmethod
    def plot_images(images, title="Image", denormalize=True, scale=1, show=True):
        """
        Helper function to plot images, reversing the transformations.
        """
        num_views = images.shape[0]
        vis_images = []

        for i in range(num_views):
            image = reverse_transform(images[i], denormalize=denormalize, IMAGENET_TRANSFORM=True)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            vis_images.append(image)

        vis_images = np.hstack(vis_images)
        vis_images = cv2.resize(vis_images, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        if show:
            cv2.imshow(title, vis_images)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return vis_images

    @staticmethod
    def plot_heatmaps(heatmaps, title="Heatmap", scale=1, show=True):
        """
        Helper function to plot combined heatmaps for multiple views
        """
        num_views = heatmaps.shape[0]
        heatmap_images = []
        for i in range(num_views):
            heatmap = heatmaps[i].clone().detach().cpu().numpy()  # Ensure heatmap is a numpy array
            heatmap = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
            heatmap_images.append(heatmap)

        # Stack images horizontally
        hm_stacked = np.hstack(heatmap_images)
        hm_stacked = cv2.resize(hm_stacked, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        hm_stacked = cv2.cvtColor(hm_stacked, cv2.COLOR_GRAY2RGB)

        if show:
            # Show the stacked image
            cv2.imshow(title, hm_stacked)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return hm_stacked

    @staticmethod
    def plot_images_with_joints(images, joints, title="Image", show=True):
        """
        Helper function to plot images with 2D joint overlays and bones.
        """
        num_views = images.shape[0]
        vis_images = []

        dark_factor = 0.7
        for i in range(num_views):
            image = reverse_transform(images[i])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            image = (image * dark_factor).astype(np.uint8)
            
            image = HandPoseVisualizer._draw_joints_on_image(image, joints[i])

            vis_images.append(image)

        vis_images = np.hstack(vis_images)
        if show:
            cv2.imshow(title, vis_images)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return vis_images
    
    @staticmethod
    def plot_images_with_vertices(images, vertices2d, title="Image", show=True):
        """
        Helper function to plot images with 2D joint overlays and bones.
        """
        num_views = images.shape[0]
        vis_images = []

        dark_factor = 0.7
        for i in range(num_views):
            image = reverse_transform(images[i])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            image = (image * dark_factor).astype(np.uint8)

            image = HandPoseVisualizer._draw_vertices_on_image(image, vertices2d[i])

            vis_images.append(image)

        vis_images = np.hstack(vis_images)
        if show:
            cv2.imshow(title, vis_images)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return vis_images

    @staticmethod
    def plot_full_images_with_joints_and_bboxes(images, joints, bboxes, title="Image", show=True):
        """
        Helper function to plot images with bounding box overlays.
        """
        num_views = images.shape[0]
        vis_images = []

        dark_factor = 0.5
        for i in range(num_views):
            image = reverse_transform(images[i])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            image = (image * dark_factor).astype(np.uint8)
            
            image = HandPoseVisualizer._draw_joints_on_image(image, joints[i])

            cv2.rectangle(image, (int(bboxes[i][0]), int(bboxes[i][1])), (int(bboxes[i][2]), int(bboxes[i][3])), (255, 0, 0), 2)  # Blue bounding box
            vis_images.append(image)

        vis_images = np.hstack(vis_images)
        if show:
            cv2.imshow(title, vis_images)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return vis_images
    
    @staticmethod
    def plot_full_images_with_vertices_and_bboxes(images, vertices2d, bboxes, title="Image", show=True):
        """
        Helper function to plot images with bounding box overlays.
        """
        num_views = images.shape[0]
        vis_images = []

        dark_factor = 0.5
        for i in range(num_views):
            image = reverse_transform(images[i])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            image = (image * dark_factor).astype(np.uint8)
            # Draw faces
            for face in HandPoseVisualizer.hand_faces:
                for fi in range(len(face)):
                    start_vertex = vertices2d[i, face[fi]]
                    end_vertex = vertices2d[i, face[(fi + 1) % len(face)]]
                    cv2.line(image, (int(start_vertex[0]), int(start_vertex[1])), (int(end_vertex[0]), int(end_vertex[1])), color=(255, 255, 255), thickness=1)

            # for v in vertices2d[i]:
            #     cv2.circle(image, (int(v[0]), int(v[1])), 1, (0, 0, 255), -1)  # Red dot for each v

            cv2.rectangle(image, (int(bboxes[i][0]), int(bboxes[i][1])), (int(bboxes[i][2]), int(bboxes[i][3])), (255, 0, 0), 2)  # Blue bounding box
            vis_images.append(image)

        vis_images = np.hstack(vis_images)
        if show:
            cv2.imshow(title, vis_images)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return vis_images

    @staticmethod
    def plot_3d_data(points, title="3D Plot", is_vertices=False, edge_color='green', show=True):
        """
        Helper function to plot 3D data.
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        if is_vertices:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='black')
            for edge in HandPoseVisualizer.hand_edges:
                joint1 = points[edge[0]]
                joint2 = points[edge[1]]
                ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], [joint1[2], joint2[2]], color=edge_color)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title(title)

        if show:
            plt.show()
        else:
            # Convert the plot to a NumPy array
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_arr = np.array(Image.open(buf))
            buf.close()
            plt.close(fig)  # Close the figure to free memory
            return img_arr

    @staticmethod
    def plot_joints_3d_predictions(points_pred, points_gt, title="3D Plot"):
        """
        Helper function to plot 3D data.
        """
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')

        # ax.scatter(points_pred[:, 0], points_pred[:, 1], points_pred[:, 2], color='blue')
        for edge in HandPoseVisualizer.hand_edges:
            joint1 = points_pred[edge[0]]
            joint2 = points_pred[edge[1]]
            ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], [joint1[2], joint2[2]], color="blue")

        # ax.scatter(points_gt[:, 0], points_gt[:, 1], points_gt[:, 2], color='green')
        for edge in HandPoseVisualizer.hand_edges:
            joint1 = points_gt[edge[0]]
            joint2 = points_gt[edge[1]]
            ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], [joint1[2], joint2[2]], linestyle='dashed', color="green")

        # # set axis labels
        # ax.set_xlabel('X axis')
        # ax.set_ylabel('Y axis')
        # ax.set_zlabel('Z axis')

        # # hide ticks
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])

        # Hide tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # # turn off axis
        # ax.axis('off')  

        # set title
        # ax.set_title(title)

        # Convert the plot to a NumPy array
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_arr = np.array(Image.open(buf))
        buf.close()
        plt.close(fig)  # Close the figure to free memory
        return img_arr

    @staticmethod
    def plot_3d_joints_and_vertices(joints, vertices, title="3D Plot"):
        """
        Helper function to plot 3D joints and vertices data.
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot vertices
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=5, color='blue', alpha=0.5)

        # Plot joints
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='red')

        # Plot edges
        for edge in HandPoseVisualizer.hand_edges:
            joint1 = joints[edge[0]]
            joint2 = joints[edge[1]]
            ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], [joint1[2], joint2[2]], color='green')

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title(title)

        plt.show()

    @staticmethod
    def visualize_3d_mesh(vertices_pred, vertices_gt=None, camera=None):
        # camera = {'eye': {'x': np.float32(0.0), 'y': np.float32(0.0), 'z': np.float32(-2.5)},
                #           'center': {'x': 0, 'y': 0, 'z': 0}, 
                #           'up': {'x': np.float32(0.0), 'y': np.float32(-1.0), 'z': np.float32(0.0)}}
        faces = HandPoseVisualizer.hand_faces
        traces = []
        if vertices_gt is not None:
            traces.append(go.Mesh3d(x=vertices_gt[:, 0], y=vertices_gt[:, 1], z=vertices_gt[:, 2],
                                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], 
                            name="gt", opacity=1, color='gray',
                            flatshading=False,))

        traces.append(go.Mesh3d(x=vertices_pred[:, 0], y=vertices_pred[:, 1], z=vertices_pred[:, 2],
                                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], 
                        name="pred", opacity=1, color='blue'))

        plot_figure = go.Figure(
                        data=traces,
                        layout=go.Layout(
                            margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                        ))
        # Update layout to remove tick labels and axis labels
        plot_figure.update_layout(
            width=256, height=256,
            scene=dict(
                xaxis=dict(showticklabels=False, title_text='', showbackground=False,),
                yaxis=dict(showticklabels=False, title_text='', showbackground=False,),
                zaxis=dict(showticklabels=False, title_text='', showbackground=False,),
                # # Optionally, hide the grid and zero lines if desired
                # xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False,
                # xaxis_zeroline=False, yaxis_zeroline=False, zaxis_zeroline=False,
            ),
            scene_camera=camera
        )
        # plot_figure.show()
        mesh = plotly.io.to_image(plot_figure, format="png")
        mesh = np.frombuffer(mesh, dtype = np.uint8)
        img = cv2.imdecode(mesh, flags = 1)
        return img
    
    @staticmethod
    def generate_mesh_from_verts(verts, camera_extr, camera_intr, background):
        import trimesh
        import pyrender
        colors = {
            'hand': [.9, .9, .9],
            'pink': [.9, .7, .7], 
            'light_blue': [0.96, 0.74, 0.65]
        }
        # print(camera_extr.shape, verts.shape, camera_extr)
        camera_extr[3] = np.array([0,0,0,1])
        # verts = np.linalg.inv(camera_extr) @ verts
        #the ratio=10 can make the rendered image be black
        ratio = 1000
        # rotation matrix to flip y-axis and z-axis signs: 180-deg rotation around x-axis
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        np_rot_x = np.reshape(np.tile(np_rot_x, [1, 1]), [3, 3])

        cv = verts @ np_rot_x
        tmesh = trimesh.Trimesh(vertices=cv / ratio, faces=HandPoseVisualizer.hand_faces, vertex_colors=colors["light_blue"])

        mesh = pyrender.Mesh.from_trimesh(tmesh)
        scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2])
        scene.add(mesh)
        
        pycamera = pyrender.IntrinsicsCamera(camera_intr[0],
                                            camera_intr[1],
                                            camera_intr[2],
                                            camera_intr[3],
                                            znear=0.1,
                                            zfar=10)

        scene.add(pycamera, pose=camera_extr)
        light = pyrender.SpotLight(color=np.ones(3), intensity=10.0)
        light_pose = camera_extr.copy()
        # Move the light to (x=0, y=0, z=1)
        light_pose[0:3, 3] = np.array([0, 0, 1])
        scene.add(light, pose=light_pose)

        light_trans = np.array([[-200., -100., -100.], [800., 10., 300.], [-500., 500., 1000.]])
        light_trans *= 0.0012
        # Add more lights
        light2 = pyrender.PointLight(color=np.array([1, 1, 1]), intensity=3)
        light_pose2 = light_pose.copy()
        light_pose2[0:3, 3] = rotateY(light_trans[0], np.radians(120))
        scene.add(light2, pose=light_pose2)

        # Add more lights
        light2 = pyrender.PointLight(color=np.array([1, 1, 1]), intensity=3)
        light_pose2 = light_pose.copy()
        light_pose2[0:3, 3] = rotateY(light_trans[1], np.radians(120))
        scene.add(light2, pose=light_pose2)

        # Add more lights
        light2 = pyrender.PointLight(color=np.array([0.7, 0.7, 0.7]), intensity=3)
        light_pose2 = light_pose.copy()
        light_pose2[0:3, 3] = rotateY(light_trans[2], np.radians(120))
        scene.add(light2, pose=light_pose2)

        r = pyrender.OffscreenRenderer(background.shape[1], background.shape[0])  # (w, h)
        recolor, depth = r.render(scene)
        
        depth_mask = depth > 0
        # Convert rendered color from RGB to BGR for OpenCV compatibility
        rendered_color_bgr = cv2.cvtColor(recolor, cv2.COLOR_RGB2BGR)

        # Blend the rendered mesh color on top of the background based on depth mask
        blended_image = np.where(depth_mask[..., None], rendered_color_bgr, background)

        return blended_image.astype("uint8"), depth