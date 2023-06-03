import torch
import cv2
import numpy as np
import torch.nn.functional as F

from .process import (Tp_map_to_line_torch,
                      convert_kp2d_pred)
from .geometry_utils import (keypoints_to_grid,
                             get_overlap_orth_line_dist
                             )


class MeanMetric(object):
    def __init__(self) -> None:
        self.counts = 0
        self.sum_value = 0.

    def update(self, value, num):
        self.counts += num
        self.sum_value += value * num

    def reset(self):
        self.counts = 0
        self.sum_value = 0.

    def compute(self):
        return self.sum_value / self.counts


def get_dist_mat(desc1: torch.Tensor, desc2: torch.Tensor, dist_type):
    eps = 1e-6
    cos_dist_mat = desc1 @ desc2.t()
    if dist_type == "cosine_dist":
        dist_mat = cos_dist_mat
    elif dist_type == "euclidean_dist":
        dist_mat = (2 - 2 * cos_dist_mat).clamp_min(eps).sqrt()
    elif dist_type == "euclidean_dist_no_sqrt":
        dist_mat = 2 - 2 * cos_dist_mat
    elif dist_type == "euclidean_dist_no_norm":
        d1 = (desc1 * desc1).sum(1, keepdim=True)
        d2 = (desc2 * desc2).sum(1, keepdim=True)
        dist_mat = (d1 - 2 * cos_dist_mat + d2.t() + eps).clamp_min(0).sqrt()
    else:
        raise NotImplementedError
    return dist_mat


def get_line_distance(ref_lines, target_lines, dis_metric="struct"):
    if dis_metric == "struct" or dis_metric == "struct_sqrt":
        # (N1, N2, 2, 2)
        if dis_metric == "struct":
            diff = ((ref_lines[:, None, :, None] -
                    target_lines[:, None]) ** 2).sum(-1)
        else:
            diff = ((ref_lines[:, None, :, None] -
                    target_lines[:, None]) ** 2).sum(-1) ** 0.5
        # (N1, N2, 1)
        diff = np.minimum(
            diff[:, :, 0, 0] + diff[:, :, 1, 1],
            diff[:, :, 0, 1] + diff[:, :, 1, 0])
    elif dis_metric == "orth":
        diff = get_overlap_orth_line_dist(
            ref_lines, target_lines, min_overlap=0.5)
    else:
        raise ValueError("{} is not supported".format(dis_metric))
    return diff


def angular_distance(segs1, segs2):
    """ Compute the angular distance (via the cosine similarity)
        between two sets of line segments. """
    # Compute direction vector of segs1
    dirs1 = segs1[:, 1] - segs1[:, 0]
    dirs1 /= (np.linalg.norm(dirs1, axis=1, keepdims=True) +
              np.finfo(dirs1.dtype).eps)
    # Compute direction vector of segs2
    dirs2 = segs2[:, 1] - segs2[:, 0]
    dirs2 /= (np.linalg.norm(dirs2, axis=1, keepdims=True) +
              np.finfo(dirs1.dtype).eps)
    # https://en.wikipedia.org/wiki/Cosine_similarity
    return np.arccos(np.minimum(1, np.abs(np.einsum('ij,kj->ik', dirs1, dirs2))))


def Ap(tp, fp):
    recall = tp
    precision = tp / np.maximum(tp + fp, 1e-8)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    return np.sum((recall[i + 1] - recall[i]) * precision[i + 1])


def calculate_line_sAp(ref_lines, target_lines, ref_scores, thresh=5, dis_metric="struct"):
    """
    Args:
        ref_lines (N, 2, 2)
        target_lines (M, 2, 2)
        ref_scores (N, 1)
    """
    if ref_lines.shape[0] == 0 and target_lines.shape[0] == 0:
        return 1.
    if ref_lines.shape[0] == 0 or target_lines.shape[0] == 0:
        return 0.
    diff = get_line_distance(ref_lines, target_lines, dis_metric)
    # (N, 1)
    # ref to target
    dist = np.min(diff, 1)
    choice = np.argmin(diff, 1)
    hit = np.zeros(len(target_lines), np.bool8)
    tp = np.zeros(len(ref_lines))
    fp = np.zeros_like(tp)
    for idx in range(len(ref_lines)):
        # ground truth line is not allowed to be matched more than once
        if (not hit[choice[idx]]) and (dist[idx] < thresh):
            tp[idx] = 1
            hit[choice[idx]] = True
        else:
            fp[idx] = 1
    sort_score_idx = np.argsort(ref_scores)[::-1]
    tps = np.cumsum(tp[sort_score_idx])
    fps = np.cumsum(fp[sort_score_idx])
    rcs = tps / target_lines.shape[0]
    prs = tps / (tps + fps + 1e-8)
    rcs = np.concatenate(([0.0], rcs, [1.0]))
    prs = np.concatenate(([0.0], prs, [0.0]))

    # Compute precision envelop
    for i in range(prs.size - 1, 0, -1):
        prs[i - 1] = max(prs[i - 1], prs[i])
    # x axis
    i = np.where(rcs[1:] != rcs[:-1])[0]
    # Compute area
    ap = np.sum((rcs[i + 1] - rcs[i]) * prs[i+1])
    return ap


def compute_mathch_score(
        batch_desc_pred1, batch_desc_pred2,
        batch_points1, batch_points2,
        grid_size, dist_type):
    b_size, _, h, w = batch_desc_pred1.shape
    img_size = (h * grid_size, w * grid_size)
    device = batch_desc_pred1.device

    # For batch
    match_score = torch.tensor(np.array([0], dtype=np.float32), device=device)
    for idx in range(b_size):
        # (N, 2)
        n_points1 = batch_points1[idx].size()[0]
        n_points2 = batch_points2[idx].size()[0]
        n_points = min(n_points1, n_points2)
        points1 = batch_points1[idx][:n_points].unsqueeze(0)
        points2 = batch_points2[idx][:n_points].unsqueeze(0)

        # Extract valid keypoints
        # TODO
        if n_points == 0:
            match_score += 1.
            continue

        # Convert the keypoints to a grid suitable for interpolation
        grid1 = keypoints_to_grid(points1, img_size)
        grid2 = keypoints_to_grid(points2, img_size)

        # Extract the descriptors
        desc_pred1 = batch_desc_pred1[idx].unsqueeze(0)
        desc_pred2 = batch_desc_pred2[idx].unsqueeze(0)
        # (B, D, n_points, 1) => (B, 1, n_points, D)
        desc1 = F.grid_sample(desc_pred1, grid1, align_corners=True).permute(
            0, 2, 3, 1).reshape(n_points, -1)
        desc1 = F.normalize(desc1, p=2, dim=1)
        desc2 = F.grid_sample(desc_pred2, grid2, align_corners=True).permute(
            0, 2, 3, 1).reshape(n_points, -1)
        desc2 = F.normalize(desc2, p=2, dim=1)

        # Distance matrix
        desc_distance = get_dist_mat(desc1, desc2, dist_type)

        # computer percentange of correct matches
        if dist_type == "cosine_dist":
            matches0 = torch.max(desc_distance, dim=1)[1]
            matches1 = torch.max(desc_distance, dim=0)[1]
        else:
            matches0 = torch.min(desc_distance, dim=1)[1]
            matches1 = torch.min(desc_distance, dim=0)[1]
        score = (matches1[matches0] == torch.arange(n_points).to(device))
        match_score += score.float().mean()

    return match_score / b_size


def val_pred_lines(images, pred_maps, target_maps, valid_mask=None, decode_mode="cd"):
    """
    Args:
    -----
    image: (b, C, H, W)
    pred_maps (b, 7, h, w)
    target_maps (b, 7, h, w)
    """
    # Some display parameters
    radius = 3
    thickness = 2

    sAp_lst = []
    pred_plot_lst = []
    target_plot_lst = []
    # Convert line
    pred_lines_lst, pred_center_lst, scores_list = Tp_map_to_line_torch(
        pred_maps,
        score_thresh=0.25,
        with_sig=True,
        image_size=images.shape[-2:],
        valid_mask=valid_mask,
        decode_mode=decode_mode,
        ncs=False)
    target_lines_lst, target_center_lst, _ = Tp_map_to_line_torch(
        target_maps,
        score_thresh=0.9,
        image_size=images.shape[-2:],
        decode_mode=decode_mode)
    for i in range(len(pred_lines_lst)):
        img = images[i].add(1.).mul(127.5).cpu().numpy().astype(
            np.uint8).transpose(1, 2, 0)
        pred_lines = pred_lines_lst[i].cpu().numpy()
        pred_center = pred_center_lst[i].cpu().numpy()
        target_lines = target_lines_lst[i].cpu().numpy()
        target_center = target_center_lst[i].cpu().numpy()
        scores = scores_list[i].cpu().numpy()
        # Compute f1socre, precision, recall of line heatmap
        scale = np.array([128.0 / img.shape[0], 128.0 / img.shape[1]])
        # f1score, precision, recall = line_F1_socre(
        #     pred_lines * scale,
        #     target_lines * scale,
        #     image_size=(128, 128))
        # f1score_lst.append(f1score)
        # precision_lst.append(precision)
        # recall_lst.append(recall)

        # Compute sAp5
        sAp = calculate_line_sAp(
            pred_lines * scale,
            target_lines * scale,
            scores,
            thresh=5)
        sAp_lst.append(sAp)

        # Only plot two img
        if i == 0 or i == len(pred_lines_lst) // 2:
            # plot line and center
            if img.shape[-1] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            pred_plot = img.copy()
            for l in (pred_lines):
                p1 = np.round(l[0, ::-1]).astype(np.int32)
                p2 = np.round(l[1, ::-1]).astype(np.int32)
                cv2.line(pred_plot, p1, p2, (0, 255, 0), thickness)
            for p in (pred_center):
                cv2.circle(pred_plot, np.round(
                    p[::-1]).astype(np.int32), radius, (0, 255, 0), radius)

            target_plot = img.copy()
            for l in (target_lines):
                p1 = np.round(l[0, ::-1]).astype(np.int32)
                p2 = np.round(l[1, ::-1]).astype(np.int32)
                cv2.line(target_plot, p1, p2, (255, 0, 0), thickness)
            for p in (target_center):
                cv2.circle(target_plot, np.round(
                    p[::-1]).astype(np.int32), radius, (255, 0, 0), radius)
            
            pred_plot_lst.append(pred_plot[None])
            target_plot_lst.append(target_plot[None])

    plot_images = {"pred_plot": np.concatenate(pred_plot_lst, axis=0).transpose(0, 3, 1, 2),
                   "gt_plot": np.concatenate(target_plot_lst, axis=0).transpose(0, 3, 1, 2)}
    scalars = {"sAp5": torch.tensor(sAp_lst).mean()}
    plot_images = {"pred_plot": torch.from_numpy(plot_images["pred_plot"]),
                   "gt_plot": torch.from_numpy(plot_images["gt_plot"])}
    return scalars, plot_images


def val_pred_points(
        images,
        points_pred_map,
        points_target_map,
        grid_size,
        detect_mode,
        cross_ratio=2,
        detect_thresh=0.5,
        top_k=800):
    """
    Args:
    -----
    image: (b, C, H, W)
    points_pred_map (b, grid_size ** 2 + 1, h / grid_size, w / grid_size)
    points_target_map (b, 1, h, w)
    """
    # Some display parameters
    radius = 2
    thickness = -1

    pred_plot_lst = []
    gt_plot_lst = []

    points = convert_kp2d_pred(
        points_pred_map, grid_size, cross_ratio, 0, 0, True)
    # plot points
    for i in [0, images.shape[0] // 2]:
        img = images[i].add(1.).mul(127.5).cpu().numpy().astype(
            np.uint8).transpose(1, 2, 0)

        if img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pred_plot = img.copy()
        for p in points[i].cpu().numpy():
            cv2.circle(pred_plot, np.round(p[[1, 0]]).astype(
                np.int32), radius, (0, 255, 0), thickness)

        gt_plot = img.copy()
        for p in points[i][:top_k].cpu().numpy():
            cv2.circle(gt_plot, np.round(p[[1, 0]]).astype(
                np.int32), radius, (255, 0, 0), thickness)
        
        pred_plot_lst.append(pred_plot[None])
        gt_plot_lst.append(gt_plot[None])
        
    scalars = {}

    plot_images = {"pred_plot": np.concatenate(pred_plot_lst, axis=0).transpose(0, 3, 1, 2),
                   "gt_plot": np.concatenate(gt_plot_lst, axis=0).transpose(0, 3, 1, 2)}

    plot_images = {"pred_plot": torch.from_numpy(plot_images["pred_plot"]),
                   "gt_plot": torch.from_numpy(plot_images["gt_plot"])}

    return scalars, plot_images
