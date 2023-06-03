import torch
import torch.nn as nn
import cv2
import numpy as np
import yaml
import h5py
import os
from typing import Optional
from tqdm import tqdm
from torch.nn import functional as F

from ..misc.geometry_utils import (keypoints_to_grid,
                                   image_grid)


def _nms(heat, kernel=3, stride=1):
    is_np = isinstance(heat, np.ndarray)
    if is_np:
        heat = torch.from_numpy(heat).unsqueeze(0)

    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=stride, padding=pad)

    keep = (hmax == heat).float()
    heat = heat * keep

    if is_np:
        heat = heat.cpu().numpy()[0]

    return heat


def simple_nms(pts, scores, H, W, top_k, nms_radius, detect_thresh=0.0, sort=True):
    """ Run a simple Non-Max-Suppression using torch.max_pool2d
    Args:
        pts: (N, 2). It should be hw format.
    """
    pts = torch.round(pts)
    if nms_radius != 0:
        pts_map = torch.zeros([1, H, W],
                            device=pts.device, dtype=torch.float32).add(-1)
        pts_map[:, pts[:, 0].to(
            torch.int64), pts[:, 1].to(torch.int64)] = scores
        if nms_radius != 0:
            pts_map_nms = _nms(
                pts_map, kernel=nms_radius * 2 + 1).squeeze(0)
        coords_nms = torch.nonzero(
            pts_map_nms >= detect_thresh, as_tuple=True)
        scores_nms = pts_map_nms[coords_nms[0], coords_nms[1]]
        pts_temp = torch.stack(
            [coords_nms[0], coords_nms[1], scores_nms],
            dim=1)
    else:
        pts_temp = torch.cat([pts, scores.unsqueeze(-1)], dim=-1)
        pts_temp = pts_temp[scores >= detect_thresh]
    if sort:
        indices = torch.argsort(pts_temp[..., 2], descending=True)
        pts_temp = pts_temp[indices]
    return pts_temp



def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T

    Algo summary: Create a grid sized HxW. Assign each corner location a 1,
    rest are zeros. Iterate through all the 1's and convert them to -1 or 0.
    Suppress points by setting nearby values to 0.

    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).

    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundary.

    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinite distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

def super_nms(prob_predictions, dist_thresh, prob_thresh=1/65, top_k=0):
    """ Non-maximum suppression adapted from SuperPoint. """
    # Iterate through batch dimension
    im_h = prob_predictions.shape[1]
    im_w = prob_predictions.shape[2]
    output_lst = []
    for i in range(prob_predictions.shape[0]):
        # print(i)
        prob_pred = prob_predictions[i, ...]
        # Filter the points using prob_thresh
        coord = np.where(prob_pred >= prob_thresh)  # HW format
        points = np.concatenate((coord[0][..., None], coord[1][..., None]),
                                axis=1)  # HW format

        # Get the probability score
        prob_score = prob_pred[points[:, 0], points[:, 1]]

        # Perform super nms
        # Modify the in_points to xy format (instead of HW format)
        in_points = np.concatenate((coord[1][..., None], coord[0][..., None],
                                    prob_score), axis=1).T
        keep_points_, keep_inds = nms_fast(in_points, im_h, im_w, dist_thresh)
        # Remember to flip outputs back to HW format
        keep_points = np.round(np.flip(keep_points_[:2, :], axis=0).T)
        keep_score = keep_points_[-1, :].T

        # Whether we only keep the topk value
        if (top_k > 0) or (top_k is None):
            k = min([keep_points.shape[0], top_k])
            keep_points = keep_points[:k, :]
            keep_score = keep_score[:k]

        # Re-compose the probability map
        output_map = np.zeros([im_h, im_w])
        output_map[keep_points[:, 0].astype(np.int),
                   keep_points[:, 1].astype(np.int)] = keep_score.squeeze()

        output_lst.append(output_map[None, ...])

    return np.concatenate(output_lst, axis=0)



def Tp_map_to_line_torch(
        tp_map,
        score_thresh=0.2,
        top_k=500,
        k_size=3,
        len_thresh=20,
        with_sig=False,
        image_size: Optional[tuple] = None,
        valid_mask=None,
):
    """
    Args
    -----
    tp_map (b, c, h, w)
    image_size: if is not None, rescale lines
    filter_point: if True, only reserve lines centerpoint  

    return
    ------
    batch_lines: (b, N, 2, 2[hw])
    batch_center_pos: (b, N, 2)
    """
    b, c, h, w = tp_map.shape
    scale = torch.tensor([h, w], device=tp_map.device, dtype=torch.float32)
    center_map = tp_map[:, 0, :, :]
    displacement_map = tp_map[:, 1:5, :, :]
    if with_sig:
        center_map = torch.sigmoid(center_map)
    if valid_mask is not None:
        valid_mask = F.interpolate(valid_mask.float(),
                                   tp_map.shape[-2:], mode="bilinear", align_corners=True)
        center_map = center_map * valid_mask.squeeze(1)

    center_max = _nms(center_map, kernel=k_size)

    batch_center_pos = []
    batch_lines = []
    batch_scores = []
    for i in range(b):
        # (topk, )
        scores, indices = torch.topk(center_max[i].reshape(-1, ), top_k)
        valid_index = torch.where(scores > score_thresh)
        scores = scores[valid_index]
        indices = indices[valid_index]
        center_y = torch.div(indices, w, rounding_mode='floor')
        center_x = torch.fmod(indices, w)
        # (N, 2)
        center_pos = torch.cat(
            (center_y.unsqueeze(-1), center_x.unsqueeze(-1)), dim=-1)
        start_points = center_pos + (displacement_map[i, :2, center_y, center_x]
                                    .permute(1, 0))
        end_points = center_pos + (displacement_map[i, 2:, center_y, center_x]
                                    .permute(1, 0))
            
        lines = torch.cat((start_points, end_points), dim=-1)
        lines = lines.reshape(-1, 2, 2)

        # Resize
        if image_size is not None:
            lines = lines / scale * \
                torch.tensor(image_size, device=lines.device)
            center_pos = center_pos / scale * \
                torch.tensor(image_size, device=lines.device)
        # Filter with length
        lines_len = torch.linalg.norm(lines[:, 0, :] - lines[:, 1, :], dim=-1)
        valid = torch.where(lines_len > len_thresh)
        lines = lines[valid]
        center_pos = center_pos[valid]
        scores = scores[valid]
        # Batch
        batch_lines.append(lines)
        batch_center_pos.append(center_pos)
        batch_scores.append(scores)

    return batch_lines, batch_center_pos, batch_scores



def convert_kp2d_pred(pred, grid_size, cross_ratio, 
                    detect_thresh=0.5, nms_radius=1, sort=False):
    batch_size, _, Hc, Wc = pred.size()
    scores = torch.sigmoid(pred[:, 0]).reshape(batch_size, -1)
    coords = convert_position_from_shift(pred[:, 1:3], grid_size, cross_ratio)
    pts = []
    if nms_radius != 0:
        coords = torch.round(coords)
    for i in range(batch_size):
        if nms_radius != 0:
            pts_map = torch.zeros([1, Hc * grid_size, Wc * grid_size],
                                  device=pred.device, dtype=torch.float32).add(-1)
            pts_map[:, coords[i, :, 0].to(
                torch.int64), coords[i, :, 1].to(torch.int64)] = scores[i]
            if nms_radius != 0:
                pts_map_nms = _nms(
                    pts_map, kernel=nms_radius * 2 + 1).squeeze(0)
            coords_nms = torch.nonzero(
                pts_map_nms >= detect_thresh, as_tuple=True)
            scores_nms = pts_map_nms[coords_nms[0], coords_nms[1]]
            pts_temp = torch.stack(
                [coords_nms[0], coords_nms[1], scores_nms],
                dim=1)
        else:
            pts_temp = torch.cat([coords[i], scores[i].unsqueeze(-1)], dim=-1)
            pts_temp = pts_temp[scores[i] >= detect_thresh]
        if sort:
            indices = torch.argsort(pts_temp[..., 2], descending=True)
            pts_temp = pts_temp[indices]
        pts.append(pts_temp)
    return pts


def convert_position_from_shift(center_shift_map, grid_size, cross_ratio):
    B, _, H, W = center_shift_map.shape
    step = (grid_size - 1) / 2.
    center_shift = torch.tanh(center_shift_map)
    center_base = image_grid(H, W).mul(grid_size) + step
    center_base = center_base.to(center_shift_map.device)
    coords = center_base + center_shift.mul(cross_ratio * step)
    new_coords_y = torch.clamp(coords[:, 0], 0, H * grid_size - 1).unsqueeze(1)
    new_coords_x = torch.clamp(coords[:, 1], 0, W * grid_size - 1).unsqueeze(1)
    new_coords = torch.cat([new_coords_y, new_coords_x], dim=1)
    return new_coords.reshape(B, 2, -1).permute(0, 2, 1)


def get_points_from_map(pts_heatmap):
    coord = np.where(pts_heatmap > 0)
    pts = np.concatenate((coord[0][..., None],
                          coord[1][..., None]),
                          axis=1)
    return pts


def extract_descriptors(pts, desc_map, image_size):
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts).to(desc_map.device)
    pts = pts.reshape(-1, 2)
    num_pts = pts.shape[0]
    grid = keypoints_to_grid(pts, image_size)
    if desc_map.ndim == 3:
        desc_map = desc_map.unsqueeze(0)
    # (1, D, n_points, 1) => (1, 1, n_points, D) ==> (n_points, D)
    pts_desc = F.grid_sample(desc_map, grid, align_corners=True).permute(
        0, 2, 3, 1).reshape(num_pts, -1)
    pts_desc = F.normalize(pts_desc, p=2, dim=1)
    return pts_desc
