import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .nets import (mobilenet_v2,
                   mobilenet_v3)

from .nets.decoder import (FpnDecoder,
                           FusedDecoder,
                           LineDecoder,
                           PointsDecoder,
                           DescriptorDecoder)

from .nets.layers import init_weight
from .line_matcher import WunschLineMatcher
from ..misc.process import (Tp_map_to_line_torch,
                            convert_kp2d_pred,
                            extract_descriptors)
from ..misc.common import time_sync


def get_model(model_cfg, mode: str = "train"):
    print("\n\n[Info]: --------Initializing model----------")
    mod = __import__(__name__, fromlist=["model_util"])
    model = getattr(mod, model_cfg["model_name"])(model_cfg)
    if mode == "train":
        model.train()
    else:
        model.eval()
    return model


class LiKL(nn.Module):
    """Full Network of LiKL"""

    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.cfg = model_cfg
        self.activation_layer = nn.Hardswish
        self.backbone = self.get_backbone()
        self.fpn_decoder = self.get_fpn_decoder()
        self.line_decoder = self.get_line_decoder()
        self.points_decoder = self.get_points_decoder()
        if "descriptor_decoder_cfg" in self.cfg:
            self.descriptor_decoder = self.get_descriptor_decoder()
        self.decoder_apply()

    def forward(self, x):
        outputs = {}
        feats = self.backbone(x)
        shard_feat = self.fpn_decoder(feats)
        line_pred = self.line_decoder(shard_feat)
        points_pred = self.points_decoder(shard_feat)
        outputs["line_pred"] = line_pred
        outputs["points_pred"] = points_pred
        if "descriptor_decoder_cfg" in self.cfg:
            desc_pred = self.descriptor_decoder(shard_feat)
            outputs["desc_pred"] = desc_pred
        return outputs

    def info_show(self):
        print("[Info]: Backbone: {}".format(self.cfg["backbone"]))
        print("[Info]: Fpn_decoder: ", self.cfg["fpn_cfg"])
        print("[Info]: Line_decoer: ", self.cfg["line_decoder_cfg"])
        if hasattr(self, "descriptor_decoder"):
            print("[Info]: Descriptor_decoder",
                  self.cfg["descriptor_decoder_cfg"])

    def get_backbone(self):
        if self.cfg["backbone"] in ["mobilenet_v3_large", "mobilenet_v3_small"]:
            backbone = mobilenet_v3(
                self.cfg["backbone"], **self.cfg["backbone_cfg"])
        elif self.cfg["backbone"] == "mobilenet_v2":
            backbone = mobilenet_v2(
                self.cfg["backbone"], **self.cfg["backbone_cfg"])
        else:
            raise ValueError("The backbone selection is not supported")
        return backbone

    def get_fpn_decoder(self):
        if self.cfg["fpn"] == "fpn":
            fpn_decoder = FpnDecoder(
                self.backbone.fpn_channel_list,
                **self.cfg["fpn_cfg"],
                activation_layer=self.activation_layer)
        elif self.cfg["fpn"] == "fused":
            fpn_decoder = FusedDecoder(
                self.backbone.fpn_channel_list,
                **self.cfg["fpn_cfg"],
                activation_layer=self.activation_layer)
        else:
            raise ValueError("The fpn decoder selection is not supported")
        return fpn_decoder

    def get_line_decoder(self):
        line_decoder = LineDecoder(
            **self.cfg["line_decoder_cfg"],
            activation_layer=self.activation_layer)
        return line_decoder

    def get_points_decoder(self):
        points_decoder = PointsDecoder(
            **self.cfg["points_decoder_cfg"],
            activation_layer=self.activation_layer)
        return points_decoder

    def get_descriptor_decoder(self):
        descriptor_decoder = DescriptorDecoder(
            **self.cfg["descriptor_decoder_cfg"],
            activation_layer=self.activation_layer)
        return descriptor_decoder

    def decoder_apply(self):
        self.fpn_decoder._init_weight()
        self.line_decoder._init_weight()
        self.points_decoder.apply(init_weight)
        if "descriptor_decoder_cfg" in self.cfg:
            self.descriptor_decoder.apply(init_weight)

    @torch.jit.ignore
    def inference(self, x, points_cfg, lines_cfg, valid_mask=None):
        # check image size, should be integer multiples of 2^5
        # if it is not a integer multiples of 2^5, padding zeros
        device = x.device
        b, c, h, w = x.shape
        h_ = math.ceil(h / 32) * 32 if h % 32 != 0 else h
        w_ = math.ceil(w / 32) * 32 if w % 32 != 0 else w
        old_img_size = [h, w]
        new_img_size = [h_, w_]
        if new_img_size != old_img_size:
            x = F.interpolate(x, new_img_size, mode="bilinear", align_corners=True)

        time_info = {
            "time_backbone": 0.0,
            "time_proc_points": 0.0,
            "time_proc_lines": 0.0}
        batch_size = x.shape[0]
        
        time_info["time_backbone"] = time_sync()
        with torch.no_grad():
            outputs = self.forward(x)
        time_info["time_backbone"] = (
            time_sync() - time_info["time_backbone"]) / batch_size
        pts_maps = outputs["points_pred"]
        line_maps = outputs["line_pred"]
        desc_map = outputs["desc_pred"]

        # Get points and desc
        batch_pts = []
        batch_pts_desc = []
        time_info["time_proc_points"] = time_sync()
        pts = convert_kp2d_pred(pts_maps,
                                points_cfg["grid_size"],
                                points_cfg["cross_ratio"],
                                points_cfg["detect_thresh"],
                                points_cfg["nms_radius"])

        for i in range(batch_size):
            pts_pred = pts[i].cpu().numpy()
            inds = np.argsort(pts_pred[:, 2])[::-1]
            pts_pred = pts_pred[inds[:points_cfg["top_k"]]]
            pts_desc = extract_descriptors(
                pts_pred[:, :2], desc_map[i], new_img_size)
            if old_img_size != new_img_size:
                pts_pred[..., :2] = pts_pred[..., :2] * \
                    np.array(old_img_size) / np.array(new_img_size)
            batch_pts.append(pts_pred)
            batch_pts_desc.append(pts_desc)
        time_info["time_proc_points"] = (
            time_sync() - time_info["time_proc_points"]) / batch_size

        # Get line
        time_info["time_proc_lines"] = time_sync()
        batch_lines, _, batch_lines_scores = Tp_map_to_line_torch(
            line_maps,
            score_thresh=lines_cfg["score_thresh"],
            len_thresh=lines_cfg["len_thresh"],
            image_size=new_img_size,
            valid_mask=valid_mask,
            with_sig=True)

        # Postprocess lines and get line points
        batch_lines_desc = []
        batch_valid_points = []
        for i in range(len(batch_lines)):
            lines = batch_lines[i].cpu().numpy()
            scores = batch_lines_scores[i].cpu().numpy()

            if lines.shape[0] == 0:
                batch_lines_desc.append([])
                batch_valid_points.append([])
                batch_lines[i] = lines
                batch_lines_scores[i] = scores
                continue
            # Get line desc
            line_points, valid_points = WunschLineMatcher.sample_line_points(
                lines,
                lines_cfg["num_samples"],
                lines_cfg["sample_min_dist"])
            lines_desc = extract_descriptors(
                line_points, desc_map[i], new_img_size)
            lines_desc = lines_desc.reshape(
                lines.shape[0], lines_cfg["num_samples"], -1)
            batch_lines_desc.append(lines_desc.cpu().numpy())
            batch_valid_points.append(valid_points)

            # Resize lines
            if old_img_size != new_img_size:
                lines = lines * np.array(old_img_size) / np.array(new_img_size)

            batch_lines[i] = lines
            batch_lines_scores[i] = scores

        time_info["time_proc_lines"] = (
            time_sync() - time_info["time_proc_lines"]) / batch_size

        # compute total time
        total_time = 0
        for k in time_info.keys():
            total_time += time_info[k]
        time_info["total_time"] = total_time

        return {
            "batch_pts": batch_pts,
            "batch_pts_desc": batch_pts_desc,
            "batch_lines": batch_lines,
            "batch_lines_score": batch_lines_scores,
            "batch_lines_desc": batch_lines_desc,
            "batch_valid_points": batch_valid_points,
            "time_info": time_info
        }
