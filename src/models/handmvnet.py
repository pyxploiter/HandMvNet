import os
import cv2
import torch
import lightning as L

from vis.visualizer import HandPoseVisualizer
from utils.camera import get_2d_joints_from_3d_joints
from datasets.utils import batch_joints_img_to_cropped_joints, batch_cropped_joints_to_joints_img

from models.losses.loss import PoseLoss
from models.metrics import PoseMetrics
from models.nets import SampleNet, JointsDecoderGCN, JointsDecoderNN
from models.utils import mask_joints, generate_centered_coordinates, generate_fov_map, soft_argmax_2d
from models.fusion import CrossAttentionFusion, CrossAttentionFusionLearnableQuery
from models.layers import make_conv_layers
from models.backbones.hrnet import HRNet
from models.backbones.resnet import ResNet18, ResNet34, ResNet50_Paper
from models.joints_to_vertices import JointsToVertices


resnet = {
    "18": ResNet18,
    "34": ResNet34,
    "50_paper": ResNet50_Paper
}

class HandMvNet(L.LightningModule):
    def __init__(self, train_params, model_params, data_params):
        super().__init__()
        self.save_hyperparameters()
        self.train_params = train_params
        self.model_params = model_params
        self.data_params = data_params

        self.debug = train_params["debug"]
        self.num_views = model_params["num_views"]
        self.batch_size = self.data_params["batch_size"]

        self.backbone_name = model_params.get("backbone", "hrnet")
        assert self.backbone_name in ["hrnet", "resnet"], "Backbone should be one of ['hrnet', 'resnet']"
        if self.backbone_name == "hrnet":
            self.backbone_type = model_params.get("backbone_type", "w40")
            self.backbone_pretrained_path = model_params.get("backbone_pretrained_path", "")
            self.backbone_channels = model_params["backbone_channels"]

            self.backbone = HRNet({
                "HRNET_TYPE": self.backbone_type,
                "PRETRAINED": os.path.join(self.backbone_pretrained_path, f"hrnetv2_{self.backbone_type}_imagenet_pretrained.pth")            
            })

            self.pose_net = torch.nn.Conv2d(
                    in_channels=self.backbone_channels[0],
                    out_channels=21,
                    kernel_size=3,
                    stride=2,
                    padding=1
                )  # Reduces spatial dims by 2x

        elif self.backbone_name == "resnet":
            self.backbone_type = model_params.get("backbone_type", "34")
            assert self.backbone_type in ["18", "34", "50_paper"], "Supports only 18, 34, 50_paper"
            self.backbone_channels = model_params["backbone_channels"]

            self.backbone = resnet[self.backbone_type]({
                "PRETRAINED": model_params.get("backbone_pretrained", True),
                "FREEZE_BATCHNORM": model_params.get("freeze_bn", False),
                "EARLY_RETURN": model_params.get("backbone_early_return", 3)
            })

            if "paper" in self.backbone_type:
                self.pose_net = make_conv_layers([self.backbone_channels[0], 512, 21], kernel=1, stride=1, padding=0, bnrelu_final=False)
            else:
                self.pose_net = torch.nn.Sequential(
                    # First layer to increase spatial dimensions from 16x16 to 32x32
                    torch.nn.ConvTranspose2d(self.backbone_channels[0], 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 32, 32)
                    torch.nn.BatchNorm2d(128),
                    torch.nn.ReLU(),
                    
                    # Second layer to reduce channels from 128 to 64, keep 32x32
                    torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, 32, 32)
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(),
                    
                    # Final layer to reduce channels to 21 for heatmaps
                    torch.nn.Conv2d(64, 21, kernel_size=3, stride=1, padding=1)  # (B, 21, 32, 32)
                )

        self.feat_dim = int(sum(self.backbone_channels)/2)      # 224 (resnet), 300 (hrnet)
        self.pos_enc_list = model_params.get("pos_enc", ["pos2d", "sin"])
        self.sinusoidal_pos = "sin" in self.pos_enc_list
        if "pos2d" in self.pos_enc_list:
            self.feat_dim += 2

        if "crop" in self.pos_enc_list:
            self.feat_dim += 10

        self.sample_nets = torch.nn.ModuleList([SampleNet([c, c//2]) for c in self.backbone_channels])

        self.fusion_layers = model_params.get("fusion_layers", 5)
        self.joints_late_fusion, self.joints_decoder = self._get_relative_joints_decoder_module()

        if not self.train_params["root_relative"]:
            self.root_late_fusion, self.root_decoder = self._get_root_joint_decoder_module()

        self.get_vertices = model_params.get("get_vertices", False)
        if self.get_vertices:
            # only configured for single gpu for now
            self.joints_to_vertices = JointsToVertices(device=self.train_params["device"], mano_dir=self.data_params["mano_models_dir"])
        # just for summary
        self.example_input_array = {"x": torch.Tensor(2, self.num_views, 3, 256, 256),
                                    "bbox": torch.Tensor(2, self.num_views, 4),
                                    "cam_params": {
                                        "intrinsic": torch.Tensor(2, self.num_views, 4),
                                        "extrinsic": torch.Tensor(2, self.num_views, 4, 4)
                                    }}
        
        ds_name = self.data_params.get("name", "dexycb")
        if ds_name == "dexycb":
            self.auc_thresh = [0.0, 0.02]
        elif ds_name == "ho3d":
            self.auc_thresh = [0.0, 0.05]
        elif ds_name == "mvhand":
            self.auc_thresh = [0.0, 0.02]
        else:
            raise NotImplementedError(f"Dataset not found: {ds_name}")

    def _get_root_joint_decoder_module(self):
        # initialize root joint decoder
        fusion = CrossAttentionFusion(feat_dim=self.feat_dim,
                                      max_tokens=self.num_views,
                                      custom_query_length=1,  # defines the number of tokens to be used as query
                                      num_layers=3,  # must be odd number
                                      drop_out=0.1)
        decoder = JointsDecoderNN(in_features=self.feat_dim)
        return fusion, decoder

    def _get_relative_joints_decoder_module(self):
        # initialize late fusion module
        if self.model_params["fusion"] == "cross_attn":
            late_fusion = CrossAttentionFusion(feat_dim=self.feat_dim,
                                               max_tokens=21*self.num_views,
                                               custom_query_length=21,  # defines the number of tokens to be used as query
                                               num_layers=self.fusion_layers,  # must be odd number
                                               drop_out=0.1)

        elif self.model_params["fusion"] == "cross_attn_learnable_query":
            late_fusion = CrossAttentionFusionLearnableQuery(feat_dim=self.feat_dim, max_tokens=21*self.num_views, drop_out=0.)
        else:
            raise NotImplementedError(f"Invalid fusion type: {self.model_params['fusion']}")

        # initialize joints decoder
        if self.model_params["use_gcn"]:
            joints_decoder = JointsDecoderGCN(in_features=self.feat_dim)
        else:
            joints_decoder = JointsDecoderNN(in_features=self.feat_dim)
        return late_fusion, joints_decoder

    def forward(self, x, bbox=None, cam_params=None):
        # x: [b, v, 3, 256, 256]
        b, v, c, h, w = x.shape
        
        # [[b*v, 40, 64, 64], [b*v, 80, 32, 32], [b*v, 160, 16, 16], [b*v, 320, 8, 8]]
        mlvl_img_feats = self.backbone(x.view(-1, c, h, w))
        # mlvl feats if it returns dict
        if isinstance(mlvl_img_feats, dict):
            """mlvl_img_feats for ResNet 34: 
                torch.Size([BN, 64, 64, 64])
                torch.Size([BN, 128, 32, 32])
                torch.Size([BN, 256, 16, 16])
            """
            # [[b*v, 256, 16, 16], [b*v, 128, 32, 32], [b*v, 64, 64, 64]]
            # reversed s.t. first element has output of last layer
            mlvl_img_feats = [v for v in reversed(mlvl_img_feats.values()) if len(v.size()) == 4]

        # for backbones that return single set of feature maps, such as resnet50_paper
        if not isinstance(mlvl_img_feats, list):
            mlvl_img_feats = [mlvl_img_feats]
        # hrnet: [b*v, 40, 64, 64] -> [b*v, 21, 32, 32]
        # resnet: [b*v, 256, 16, 16] -> [b*v, 21, 32, 32]
        joint_hms = self.pose_net(mlvl_img_feats[0])
        # [b*v, 21, 32, 32] -> [b*v, 21, 2] 
        joint_coords = soft_argmax_2d(joint_hms)
        # hrnet: [[b*v, 21, 40/2], [b*v, 21, 80/2], [b*v, 21, 160/2], [b*v, 21, 320/2]]
        # resnet: [[b*v, 21, 256/2], [b*v, 21, 128/2], [b*v, 21, 64/2]]
        mlvl_sampled_feats = [net(mlvl_img_feats[i], joint_coords) for i, net in enumerate(self.sample_nets)]
        # [b*v, 21, 300] | [b*v, 21, 224]
        img_sampled_feats = torch.cat(mlvl_sampled_feats, dim=-1)

        if "pos2d" in self.pos_enc_list:
            # [b*v, 21, 302] | [b*v, 21, 226]
            img_sampled_feats = torch.cat([img_sampled_feats, joint_coords], dim=2)
        
        # [b, v, 21, 32, 32]
        joint_hms = joint_hms.view(-1, self.num_views, joint_hms.shape[1], joint_hms.shape[2], joint_hms.shape[3])
        # [b, v, 21, 2]
        joint_coords = joint_coords.view(-1, self.num_views, joint_coords.shape[1], joint_coords.shape[2])
        
        if self.debug:
            print("----------------------------------")
            # print("img feats:", img_feats.shape, img_feats.device)
            print("joint_coords:", joint_coords.shape, joint_coords.device)
            print("img_sampled_feats:", img_sampled_feats.shape, img_sampled_feats.device)
            print("joint_hms:", joint_hms.shape, joint_hms.device)

        if "crop" in self.pos_enc_list:
            bboxes = bbox.view(-1, 4)  # [b*v, 4]
            # [b*v, 5, 2]
            points = torch.stack([
                        bboxes[:, 0], bboxes[:, 1],
                        bboxes[:, 0], bboxes[:, 3],
                        bboxes[:, 2], bboxes[:, 1],
                        bboxes[:, 2], bboxes[:, 3],
                        (bboxes[:, 0] + bboxes[:, 2]) / 2, (bboxes[:, 1] + bboxes[:, 3]) / 2
                    ], dim=1).view(bboxes.shape[0], 5, 2).to(torch.float32)

            intrinsic = cam_params["intrinsic"].view(-1, 4).to(torch.float32)  # [b*v, 4]
            cc = generate_centered_coordinates(points, intrinsic[:, 2].unsqueeze(1), intrinsic[:, 3].unsqueeze(1))  # [b*v, 5, 2]
            fov = generate_fov_map(cc, intrinsic[:, 0].unsqueeze(1), intrinsic[:, 1].unsqueeze(1))  # [b*v, 5, 2]
            # [b*v, 5, 2] -> [b*v, 10] -> [b*v, 21, 10]
            expanded_fov = fov.flatten(start_dim=-2).unsqueeze(1).expand(-1, 21, -1)
            # [b*v, 21, 514+5*d]
            img_sampled_feats = torch.cat((img_sampled_feats, expanded_fov), dim=2)

        # [b*v, 21, self.feat_dim] -> [b, v*21, self.feat_dim]
        img_sampled_feats = img_sampled_feats.view(-1, self.num_views*img_sampled_feats.shape[1], img_sampled_feats.shape[2])
        # [b, v*21, self.feat_dim] -> [b, 21, self.feat_dim]
        out_joint_feats = self.joints_late_fusion(img_sampled_feats, add_pos=self.sinusoidal_pos)
        # [b, 21, self.feat_dim] -> [b, 21, 3]
        joints_cam = self.joints_decoder(out_joint_feats) 

        if self.debug:
            print("----------------------------------")
            # print("joints_coords:", joint_coords.shape, joint_coords.device)
            print("joints_cam:", joints_cam.shape, joint_coords.device)

        if not self.train_params["root_relative"]:
            abs_joint_coords = batch_cropped_joints_to_joints_img(joint_coords.view(-1, 21, 2), bbox.view(-1, 4), self.data_params["image_size"])  # [b*v, 21, 2]
            img_sampled_feats_abs_coords = torch.cat([img_sampled_feats, abs_joint_coords], dim=2)  # [b*v, 21, 514]
            root_img_feats = img_sampled_feats_abs_coords[:, 0, :].view(-1, self.num_views, img_sampled_feats_abs_coords.shape[-1])  # [b, v, 514]
            out_root_feats = self.root_late_fusion(root_img_feats)  # [b, 1, 514]
            root_joint = self.root_decoder(out_root_feats)  # [b, 1, 3]

            if self.debug:
                print("----------------------------------")
                print("abs_joint_coords:", abs_joint_coords.shape, abs_joint_coords.device)
                print("img_sampled_feats_abs_coords:", img_sampled_feats_abs_coords.shape, img_sampled_feats_abs_coords.device)
                print("root_img_feats:", root_img_feats.shape, root_img_feats.device)
                print("out_root_feats:", out_root_feats.shape, out_root_feats.device)
                print("root_joint:", root_joint.shape, root_joint.device)

        # rescale 2d joints to image shape
        joint_coords = joint_coords * self.data_params["image_size"]/self.data_params["heatmap_size"]

        if not self.train_params["root_relative"]:
            return {
                    "joints_crop_img": joint_coords,  # [b, v, 21, 2]
                    "joints_cam": joints_cam,  # [b, 21, 3], in meters
                    "root_joint": root_joint,  # [b, 1, 3], in meters
                    "heatmap": joint_hms,  # [b, v, 21, 32, 32]
                }
        else:
            return {
                "joints_crop_img": joint_coords,  # [b, v, 21, 2]
                "joints_cam": joints_cam,  # [b, 21, 3], in meters
                "heatmap": joint_hms,  # [b, v, 21, 32, 32]
            }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams["train_params"]["lr"],
                                      weight_decay=self.hparams["train_params"]["weight_decay"])

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.hparams["train_params"]["lr_scheduler"]["milestones"],
                                                         gamma=self.hparams["train_params"]["lr_scheduler"]["gamma"])

        return [optimizer], [scheduler]

    def _calculate_loss(self, out, inputs, cam_params, mode="train"):
        # Initialize a dictionary to store losses
        losses = {}

        # Precompute common terms
        heatmap_loss_weight = self.train_params["loss_weights"]["heatmap"]
        joints_crop_img_weight = self.train_params["loss_weights"]["joints_2d"]
        joints_cam_weight = self.train_params["loss_weights"]["joints_3d"]

        # Calculate heatmap loss
        losses['heatmap_loss'] = PoseLoss.mse_loss(preds=out["heatmap"],
                                                labels=inputs["heatmap"],
                                                weight=heatmap_loss_weight)

        # Calculate 2D joints loss (with or without masking)
        if "joints_img_mask" in inputs:
            joints_img_mask = inputs["joints_img_mask"]
            preds_joints_2d = mask_joints(out["joints_crop_img"], joints_img_mask) if self.train_params["mask_invisible_joints"] else out["joints_crop_img"]
            labels_joints_2d = mask_joints(inputs["joints_crop_img"], joints_img_mask) if self.train_params["mask_invisible_joints"] else inputs["joints_crop_img"]
        else:
            preds_joints_2d = out["joints_crop_img"]
            labels_joints_2d = inputs["joints_crop_img"]
                                          
        losses['joints_2d_loss'] = PoseLoss.l1_loss(preds=preds_joints_2d,
                                                    labels=labels_joints_2d,
                                                    weight=joints_crop_img_weight)

        # Calculate 3D joints loss
        losses['joints_3d_loss'] = PoseLoss.l1_loss(preds=out["joints_cam"],
                                                    labels=inputs["joints_cam"],
                                                    weight=joints_cam_weight)

        # Calculate root joint loss if not root relative
        if not self.train_params["root_relative"]:
            losses['root_3d_loss'] = PoseLoss.l1_loss(preds=out["root_joint"],
                                                    labels=inputs["root_joint"],
                                                    weight=joints_cam_weight)
        else:
            losses['root_3d_loss'] = 0.

        # Initialize projected losses
        losses['g2d_loss'] = 0.
        losses['p2d_loss'] = 0.
        # Check if projected loss calculation is required
        if "g2d" in self.train_params["loss_weights"]:
            root_joint = inputs["root_joint"] if self.train_params["root_relative"] else out["root_joint"]
            projected_joints_img = get_2d_joints_from_3d_joints(out["joints_cam"] + root_joint,
                                                                inputs["root_idx"][0],
                                                                cam_params["intrinsic"],
                                                                cam_params["extrinsic"])

            # Adjust projected joints to crop image space
            bboxes = inputs["bboxes"]
            projected_joints_crop_img = batch_joints_img_to_cropped_joints(projected_joints_img.view(-1, 21, 2), bboxes.view(-1, 4))
            projected_joints_crop_img = projected_joints_crop_img.view(-1, self.num_views, 21, 2)
            out["projected_joints_crop_img"] = projected_joints_crop_img

            # Compute projected 2D losses
            losses['g2d_loss'] = PoseLoss.l1_loss(preds=projected_joints_crop_img,
                                                  labels=inputs["joints_crop_img"],
                                                  weight=self.train_params["loss_weights"]["g2d"])
            losses['p2d_loss'] = PoseLoss.l1_loss(preds=projected_joints_crop_img,
                                                  labels=out["joints_crop_img"],
                                                  weight=self.train_params["loss_weights"]["p2d"])

        # Final loss sum
        losses["loss"] = sum(losses.values())

        # Logging losses
        for key, value in losses.items():
            self.log(f'{mode}/{key}', value, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        return losses["loss"]

    def _get_metrics(self, pred_pts, target_pts):
        # pred_pts: [b, num_pts, 3]
        # target_pts: [b, num_pts, 3]

        scale = 1000
        # Calculate AUC, normalized AUC, PCK values at each threshold, and thresholds
        auc, norm_auc, pck_values, auc_thresholds = PoseMetrics.pck_auc(preds=pred_pts,
                                                                        labels=target_pts,
                                                                        min_threshold=self.auc_thresh[0],
                                                                        max_threshold=self.auc_thresh[1],
                                                                        steps=20)
        
        mpjpe = PoseMetrics.mpjpe(pred_pts, target_pts)*scale
        pa_mpjpe = PoseMetrics.pa_mpjpe(pred_pts, target_pts)*scale
        
        return mpjpe, pa_mpjpe, auc, norm_auc, pck_values, auc_thresholds

    def _calculate_mpjpe(self, out, inputs, mode="train"):
        # Mask joints if necessary
        joints_crop_img_pred = mask_joints(out["joints_crop_img"], inputs["joints_img_mask"]) if "joints_img_mask" in inputs else out["joints_crop_img"]
        joints_crop_img_gt = mask_joints(inputs["joints_crop_img"], inputs["joints_img_mask"]) if "joints_img_mask" in inputs else inputs["joints_crop_img"]

        out_metrics = {}
        # Compute 3D joint-based metrics
        mpjpe, pa_mpjpe, auc_j, norm_auc_j, pck_values_j, _ = self._get_metrics(out["joints_cam"], inputs["joints_cam"])

        # 2D MPJPE metric
        out_metrics.update({
            f"{mode}_mpjpe2d": PoseMetrics.mpjpe(joints_crop_img_pred, joints_crop_img_gt),
            f"{mode}_mpjpe": mpjpe,
            f"{mode}_pa_mpjpe": pa_mpjpe,
            f"{mode}_pck_j": pck_values_j,
            f"{mode}_auc_j": auc_j,
            f"{mode}_norm_auc_j": norm_auc_j,
        })

        # Vertices processing if required
        if self.get_vertices:
            b = out["joints_cam"].shape[0]
            out["vertices"] = torch.zeros((b, 778, 3), device=self.device, dtype=torch.float32)  # Directly initialize on device

            # Calculate vertices for each sample in batch
            for s_idx in range(b):
                input_joints = out["joints_cam"][s_idx].detach().cpu().numpy() * 1000
                vertices = self.joints_to_vertices(input_joints)
                out["vertices"][s_idx] = torch.from_numpy(vertices).float().to(self.device)

            # Compute vertex-based metrics
            mpvpe, pa_mpvpe, auc_v, norm_auc_v, pck_values_v, _ = self._get_metrics(out["vertices"] / 1000., inputs["vertices"] / 1000.)
            out_metrics.update({
                f"{mode}_mpvpe": mpvpe,
                f"{mode}_pa_mpvpe": pa_mpvpe,
                f"{mode}_pck_v": pck_values_v,
                f"{mode}_auc_v": auc_v,
                f"{mode}_norm_auc_v": norm_auc_v
            })

        # World-MPJPE if not root-relative
        if not self.train_params["root_relative"]:
            w_mpjpe = PoseMetrics.mpjpe(out["joints_cam"] + out["root_joint"],
                                        inputs["joints_cam"] + inputs["root_joint"]) * 1000  # Convert to mm
            out_metrics[f"{mode}_w_mpjpe"] = w_mpjpe
            self.log(f'{mode}_w_mpjpe', w_mpjpe, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        # Logging the metrics
        for key, value in out_metrics.items():
            prog_bar = (key == f"{mode}_mpjpe")
            if "pck" not in key:
                self.log(key, value, prog_bar=prog_bar, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        return out_metrics

    def _visualize_output(self, out, batch, mode="train"):
        epoch_idx = self.trainer.current_epoch
        sample_idx = torch.randint(0, self.batch_size, (1,)).item()

        # print(epoch_idx, self.train_params["vis_every_n_epochs"], sample_idx, mode, )
        if (epoch_idx + 1) % self.train_params["vis_every_n_epochs"] == 0:
            # plot 2d joints on image
            joints2d_vis = HandPoseVisualizer.plot_images_with_joints(batch["data"]["rgb"][sample_idx],
                                                                      out["joints_crop_img"][sample_idx],
                                                                      show=False)
            cv2.imwrite(f"{self.train_params['vis_dir']}/{mode}/joints2d_{epoch_idx}.png", joints2d_vis)
            # plot projected 2d joints on image
            if "projected_joints_crop_img" in out:
                proj_joints2d_vis = HandPoseVisualizer.plot_images_with_joints(batch["data"]["rgb"][sample_idx],
                                                                               out["projected_joints_crop_img"][sample_idx],
                                                                               show=False)
                cv2.imwrite(f"{self.train_params['vis_dir']}/{mode}/projected_joints2d_{epoch_idx}.png", proj_joints2d_vis)

    def training_step(self, batch, batch_idx):
        # out: {joints_crop_img, joints_cam, heatmap}
        inputs = batch["data"]
        x = inputs["rgb"]
        bbox = inputs["bboxes"]
        cam_params = batch["cam_params"]
        out = self.forward(x, bbox, cam_params)

        # for numeric stability, using meters as unit
        inputs["joints_cam"] /= 1000
        inputs["root_joint"] /= 1000

        # Calculate losses and log them
        loss = self._calculate_loss(out, inputs, cam_params, mode="train")
        # Calculate MPJPE for logging
        out_metrics = self._calculate_mpjpe(out, inputs, mode="train")
        # visualize output
        if batch_idx == 0:
            self._visualize_output(out, batch, mode="train")

        return {
            "loss": loss,
            "metrics": out_metrics
        }

    def validation_step(self, batch, batch_idx):
        # out: {joints_crop_img, joints_cam, heatmap}
        inputs = batch["data"]
        x = inputs["rgb"]
        bbox = inputs["bboxes"]
        cam_params = batch["cam_params"]
        out = self.forward(x, bbox, cam_params)

        # for numeric stability, using meters as unit
        inputs["joints_cam"] /= 1000
        inputs["root_joint"] /= 1000

        # Calculate losses and log them
        loss = self._calculate_loss(out, inputs, cam_params, mode="val")
        # Calculate MPJPE for logging
        out_metrics = self._calculate_mpjpe(out, inputs, mode="val")
        # visualize output
        if batch_idx == 0:
            self._visualize_output(out, batch, mode="val")

        return {
            "loss": loss,
            "metrics": out_metrics
        }

    def test_step(self, batch, batch_idx):
        # out: {joints_crop_img, joints_cam, heatmap}
        inputs = batch["data"]
        x = inputs["rgb"]
        bbox = inputs["bboxes"]
        cam_params = batch["cam_params"]
        out = self.forward(x, bbox, cam_params)

        # for numeric stability, using meters as unit
        inputs["joints_cam"] /= 1000
        inputs["root_joint"] /= 1000

        # Calculate losses and log them
        loss = self._calculate_loss(out, inputs, cam_params, mode="test")
        # Calculate MPJPE for logging
        out_metrics = self._calculate_mpjpe(out, inputs, mode="test")
        # visualize output
        if batch_idx == 0:
            self._visualize_output(out, batch, mode="test")

        return {
            "loss": loss,
            "metrics": out_metrics
        }
