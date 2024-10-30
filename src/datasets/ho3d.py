import os

import torch
import braceexpand
import numpy as np
import webdataset as wds
from typing import List, Union
import lightning as L
import torchvision.transforms as transforms

from datasets.utils import *
from datasets.augment import SampleAugmentor


class HO3DSamplePreprocessor:
    def __init__(self, config, subset):
        self.config = config
        self.subset = subset
        self.AUGMENT = self.config.get("augment", False)
        self.NOISE_3D = self.config.get("noise_3d", False)
        self.IS_TRAIN_SET = subset == "train"

        self.total_views = 5
        self.selected_views = np.array(self.config.get("selected_views", range(self.total_views)))
        self.num_views = len(self.selected_views)
        self.input_res = (480, 640)  # (h, w)
        self.scale = 1000            # convert keypoints to milimeters

        self.full_img_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                ])

        self.img_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((self.config["image_size"], self.config["image_size"]), antialias=True),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                            ])

        self.hm_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((self.config["heatmap_size"], self.config["heatmap_size"]), antialias=True),
                            ])


        self.rgb_augmentor = SampleAugmentor()
        
        
    def __call__(self, sample):
        # Sample structure breakdown:
        # --------------------------------------------
        # sample = dict_keys(['__key__', '__url__', 'image_0.jpg', '__local_path__', 'image_1.jpg', 
        #                      'image_2.jpg', 'image_3.jpg', 'image_4.jpg', 'label.pyd'])
        # 
        # Key attributes in the sample:
        # - `__key__`: A unique identifier for the sample (e.g., '000002187')
        # - `__url__`: Path to the .tar file containing the sample
        # - `__local_path__`: Local path to the .tar file 
        # - `image_0.jpg, image_1.jpg, ...`: Individual image files corresponding to multiple camera views
        # - `label.pyd`: Contains metadata and annotations related to the sample (explained below)
        # 
        # Label file (`label.pyd`) structure:
        # --------------------------------------------
        # labels.pyd = dict_keys(['sample_idx', 'cam_extr', 'cam_serial', 'idx', 'bbox_center', 
        #                         'bbox_scale', 'cam_intr', 'joints_2d', 'joints_3d', 'verts_3d', 
        #                         'joints_vis', 'mano_pose', 'mano_shape', 'image_path', 'raw_size'])
        #
        # Key attributes in the label file (each attr is list of num_cams, only mentioning one element of list):
        # - `sample_idx`: (int) The sample's unique identifier
        # - `cam_extr`: (numpy.ndarray, shape (4, 4))
        # - `cam_serial`: (str) Serial number of the camera
        # - `idx`: (int) Index for the specific sample
        # - `bbox_center`: (numpy.ndarray, shape (2,)) The center coordinates of the bounding box
        # - `bbox_scale`: (numpy.float64) The scaling factor of the bounding box
        # - `cam_intr`: (numpy.ndarray, shape (3, 3)) The camera's intrinsic matrix (3x3 matrix)
        # - `joints_2d`: (numpy.ndarray, shape (21, 2)) 2D joint locations of the hand
        # - `joints_3d`: (numpy.ndarray, shape (21, 3)) 3D joint locations of the hand in the camera coordinate space
        # - `verts_3d`: (numpy.ndarray, shape (778, 3)) 3D vertices of the hand mesh
        # - `joints_vis`: (numpy.ndarray, shape (21,)) Visibility flag for each joint (1 if visible, 0 if not)
        # - `mano_pose`: (numpy.ndarray, shape (48,)) Pose parameters of the MANO hand model
        # - `mano_shape`: (numpy.ndarray, shape (10,)) Shape parameters of the MANO hand model
        # - `image_path`: (str) Path to the corresponding image
        # - `raw_size`: (numpy.ndarray, shape (2,)) Original size of the image before any processing
        #
        # Additional Notes:
        # ---------------------------------------------
        # - `sample["__key__"]` and `labels.pyd["sample_idx"]` do not match.
        # - Each attribute in `labels.pyd` contains data for multiple cameras. The length of each value 
        #   corresponds to the number of camera views available in the dataset.

        root_idx = 0
        labels = sample["label.pyd"]

        # ------------ Camera Params -------------
        extrinsics = np.array(labels["cam_extr"])       # (v, 4, 4)
        intrinsics_matrix = np.array(labels["cam_intr"])  # (v, 3, 3)
        # [fx, fy, cx, cy]
        intrinsics = np.array([[mat[0, 0], mat[1, 1], mat[0, 2], mat[1, 2]] for mat in intrinsics_matrix])  # (v, 4)

        # ------------ BBOX -------------
        bboxes = batch_center_scale_to_box(np.array(labels["bbox_center"]), np.array(labels["bbox_scale"]))
        # restricting bbox to image boundaries
        cropped_bboxes = np.stack([bbox_to_cropped_bbox(bbox, self.input_res) for bbox in bboxes])
        
        # ------------ MANO Params -------------
        all_mano_pose = np.array(labels["mano_pose"])       # (v, 48)
        all_mano_shape = np.array(labels["mano_shape"])     # (v, 10)
        mano_pose = all_mano_pose[root_idx]                 # (48,)
        mano_shape = all_mano_shape[root_idx]               # (10,)

        # ------------ Keypoints -------------
        joints_img = np.array(labels["joints_2d"])      # (v, 21, 2)
        joints_crop_img = batch_joints_img_to_cropped_joints(joints_img, bboxes)    # (v, 21, 2)
        joints_3d = np.array(labels["joints_3d"]) * self.scale      # (v, 21, 3)
        verts_3d = np.array(labels["verts_3d"]) * self.scale        # (v, 778, 3)

        all_root_joints = joints_3d[:, 0, :][:, None, :]    # (v, 1, 3)
        all_joints_cam = joints_3d - all_root_joints        # (v, 21, 3)
        all_vertices = verts_3d - all_root_joints           # (v, 778, 3)
        
        root_joint = all_root_joints[root_idx]          # (1, 3)
        joints_cam = all_joints_cam[root_idx]           # (21, 3)
        vertices = all_vertices[root_idx]               # (778, 3)

        # ------------ Joints Mask -------------
        joints_img_vis_mask = np.array(labels["joints_vis"])     # (v, 21) - False for invisible, True for visible
        invisible_joints_img_mask = joints_img_vis_mask == 0     # (v, 21) - True for invisible, False for visible

        # ------------ RGB Images -------------
        # sort image keys
        sorted_keys = sorted([k for k in sample.keys() if k.startswith("image")], key=lambda x: int(x.split('_')[1].split('.')[0]))
        full_rgb = np.stack([sample[k] for k in sorted_keys], axis=0)  # (v, h, w, 3)
        rgb = []
        for i in range(self.num_views):
            # check if all joints are invisible
            if not np.any(joints_img_vis_mask[i]):
                # create black images if no joint is visible
                transformed_img = self.img_transform(np.zeros([10, 10, 3], dtype=np.uint8))
            else:
                # crop image and pad with zeros if bbox is outside image boundary
                cropped_img = crop_and_pad_image(full_rgb[i], bboxes[i])

                if self.IS_TRAIN_SET and self.AUGMENT:
                    # augment images
                    transformed_img = self.img_transform(self.rgb_augmentor(cropped_img))
                else:
                    transformed_img = self.img_transform(cropped_img)
            rgb.append(transformed_img)

        full_rgb = torch.stack([self.full_img_transform(img) for img in full_rgb], dim=0)
        rgb = torch.stack(rgb)

        # ------------ Heatmaps -------------
        heatmaps = []
        for i in range(self.num_views):
            hms = []
            for j in range(21):
                hm = np.zeros((self.config["image_size"], self.config["image_size"]))
                hm = generate_heatmap(hm, joints_crop_img[i][j], sigma=2)
                hms.append(self.hm_transform(hm))

            heatmaps.append(torch.cat(hms))

        heatmaps = torch.stack(heatmaps).to(dtype=torch.float32)

        # ------------ Complete Sample -------------
        ret_sample = {
            "mv_sample_id": os.path.join(sample["__url__"], sample["__key__"]),
            "selected_views": self.selected_views,
            "image_paths": labels["image_path"],
            "sample_idx": labels["sample_idx"],
            "idx": labels["idx"],
            "cam_params": {
                "intrinsic": intrinsics,                # (v, 4), [fx, fy, cx, cy]
                "intrinsic_mat": intrinsics_matrix,     # (v, 3, 3)
                "extrinsic": extrinsics                 # (v, 4, 4)
            },
            "data": {
                "full_rgb": full_rgb,                   # (v, 3, h, w)
                "rgb": rgb,                             # (v, 3, 256, 256)
                "joints_crop_img": joints_crop_img,     # (v, 21, 2)
                "joints_img": joints_img,               # (v, 21, 2)
                "heatmap": heatmaps,                    # (v, j, 32, 32)
                "joints_img_mask": invisible_joints_img_mask, # (v, 21): 1 = invisible, 0 = visible
                # "all_mano_pose": all_mano_pose,         # (v, 48)
                # "all_mano_shape": all_mano_shape,       # (v, 10)
                "joints_cam": joints_cam,               # (21, 3)
                "root_joint": root_joint,               # (3, )
                "vertices": vertices,                   # (778, 3)
                "all_joints_cam": all_joints_cam,       # (v, 21, 3)
                "all_root_joints": all_root_joints,     # (v, 3)
                "all_vertices": all_vertices,           # (v, 778, 3)
                "mano_pose": mano_pose,                 # (48,), of root_camera
                "mano_shape": mano_shape,               # (10,), of root_camera
                "cropped_bboxes": cropped_bboxes,       # (v, 4)
                "bboxes": bboxes,                       # (v, 4)
                "root_idx": root_idx                    # (1,)
            }
        }
        
        # # visualize sample keys and value shapes
        # for key in ret_sample.keys():
        #     if isinstance(ret_sample[key], dict):
        #         for sub_key in ret_sample[key].keys():
        #             print(key, sub_key, ret_sample[key][sub_key].shape)
        #     elif not isinstance(ret_sample[key], str):
        #         print(key, len(ret_sample[key]))
        #     else:
        #         print(key, ret_sample[key])

        return ret_sample

class HO3DMultiview:
    def __init__(self, config):
        self.name = type(self).__name__
        self.cfg = config
        self.add_val_to_train = self.cfg.get("add_val_to_train", False)
        self.data_urls = {
            "train": os.path.join(self.cfg["dataset_dir"], "HO3D_mv_train-{000000..00008}.tar"),
            "val": os.path.join(self.cfg["dataset_dir"], "HO3D_mv_train-{000000..00008}.tar"),
            "test": os.path.join(self.cfg["dataset_dir"], "HO3D_mv_test-{000000..00002}.tar"),
        }

    def expand_urls(self, urls: Union[str, List[str]]):
        if isinstance(urls, str):
            urls = [urls]
        urls = [u for url in urls for u in braceexpand.braceexpand(
                                                os.path.expanduser(os.path.expandvars(url))
                                            )]
        return urls
    
    def get_dataset(self, data_split="train"):
        assert data_split in ["train", "test", "val"], f"{self.name} unsupported data split {data_split}"
        
        urls = self.expand_urls(self.data_urls[data_split])

        if self.add_val_to_train:
            print("[warning] Using train+val split as training set.")
            # for training, use all urls
            # for validation, use first and last shard
            if data_split == "val":
                urls = [urls[0], urls[-1]]
        else:
            if data_split == "train":
                urls = urls[1:-1]
            elif data_split == "val":
                urls = [urls[0], urls[-1]]

        dataset = wds.WebDataset(urls=urls,
                                 nodesplitter=wds.split_by_node,
                                 workersplitter=wds.split_by_worker,
                                 shardshuffle=data_split == "train",
                                 resampled=data_split == "train")

        if data_split == "train":
            print(f"[Dangerous] Resampled={data_split == 'train'}, Mode={data_split}")
            dataset = dataset.shuffle(500)

        dataset = dataset.decode('rgb8')
        my_processor = HO3DSamplePreprocessor(self.cfg, subset=data_split)
        dataset = dataset.map(my_processor)

        return dataset


class HO3DDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.add_val_to_train = self.cfg.get("add_val_to_train", False)
        ho3d = HO3DMultiview(self.cfg)

        if self.add_val_to_train:
            self.train_samples = 9087
        else:
            self.train_samples = 7718    # 9087
        self.test_samples = 2706
        self.val_samples = 1369      # 2706
        self.train_set = ho3d.get_dataset(data_split="train")
        self.val_set = ho3d.get_dataset(data_split="val")
        self.test_set = ho3d.get_dataset(data_split="test")

    def train_dataloader(self):
        return wds.WebLoader(self.train_set,
                        batch_size=self.cfg["batch_size"],
                        num_workers=self.cfg["num_workers"],
                        pin_memory=True
                    ).with_epoch(self.train_samples // self.cfg["batch_size"]).shuffle(self.cfg["batch_size"]*2)

    def val_dataloader(self):
        return wds.WebLoader(self.val_set,
                        batch_size=self.cfg["batch_size"],
                        num_workers=1,
                        pin_memory=True
                    ).with_epoch(self.val_samples // self.cfg["batch_size"])

    def test_dataloader(self):
        return wds.WebLoader(self.test_set,
                        batch_size=self.cfg["batch_size"],
                        num_workers=self.cfg["num_workers"],
                        pin_memory=True
                    ).with_epoch(self.test_samples // self.cfg["batch_size"])

    def predict_dataloader(self):
        return wds.WebLoader(self.test_set,
                        batch_size=self.cfg["batch_size"],
                        num_workers=self.cfg["num_workers"],
                        pin_memory=True
                    ).with_epoch(self.test_samples // self.cfg["batch_size"])


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from vis.visualizer import HandPoseVisualizer
    config = {
        "data": {
            "name": "ho3d",
            "batch_size": 1,
            "dataset_dir": "data/ho3d",
            "mask_size": 32,
            "depth_size": 32,
            "heatmap_size": 32,
            "image_size": 256,
            "mano_models_dir": "src/mano",
            "num_workers": 0,
            "augment": True,

        }
    }

    dm = HO3DDataModule(config["data"])
    dataset = dm.test_set

    print("dataloading")
    # dataloader = dm.train_dataloader()
    dataloader = dm.test_dataloader()
    for batch_idx, batch in enumerate(dataloader):
        vis_imgs = {
            "full": [],
            "crop": []
        }
        visualizer = HandPoseVisualizer(batch)

        # visualizer.visualize_3d_joints_and_vertices()
        # visualizer.visualize_3d_absolute_joints_and_vertices()

        vis_imgs["full"].append(visualizer.visualize_full_rgb_image()[:,:,::-1])
        vis_imgs["full"].append(visualizer.visualize_joints_2d_on_full_image()[:,:,::-1])
        vis_imgs["full"].append(visualizer.visualize_projected_joints_2d_on_full_image()[:,:,::-1])
        vis_imgs["full"].append(visualizer.visualize_projected_vertices_2d_on_full_image()[:,:,::-1])

        vis_imgs["crop"].append(visualizer.visualize_rgb_image()[:,:,::-1])
        vis_imgs["crop"].append(visualizer.visualize_joints_2d_on_cropped_image()[:,:,::-1])
        vis_imgs["crop"].append(visualizer.visualize_combined_heatmaps())
        
        # vis_full = np.vstack(vis_imgs["full"])
        # fig = plt.figure(figsize=(10, 10))
        # plt.imshow(vis_full)
        # plt.axis("off")
        # plt.show()

        vis_crop = np.vstack(vis_imgs["crop"])
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(vis_crop)
        plt.axis("off")
        plt.show()
        # print(batch["mv_sample_id"][0].split("/")[-1], batch["image_paths"])
        if batch_idx >= 10: break