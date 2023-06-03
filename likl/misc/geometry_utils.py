"""
Code are adapted from https://github.com/cvg/SOLD2/blob/main/sold2/
"""
import torch
import numpy as np
import torch.nn.functional as F
from functools import lru_cache

# Warp a list of points using a homography
def warp_points(points, homography, points_format="hw", pre_normalized=False, shape=None):
    assert points_format in ["xy", "hw"]
    if points_format == "hw":
        # Convert to homogeneous and in xy format
        new_points = np.concatenate([points[..., [1, 0]],
                                    np.ones_like(points[..., :1])], axis=-1)
    else:
        new_points = np.concatenate([points,
                                    np.ones_like(points[..., :1])], axis=-1)
    if pre_normalized:
        new_points[..., 0] = new_points[..., 0] / (float(shape[1]-1) / 2.) - 1
        new_points[..., 1] = new_points[..., 1] / (float(shape[0]-1) / 2.) - 1
    # Warp
    new_points = (homography @ new_points.T).T
    new_points = new_points[..., :2] / new_points[..., 2:]
    if pre_normalized:
        # Rescale to original size
        new_points[..., 0] = (new_points[..., 0] + 1) * \
            (float(shape[1]-1) / 2.)
        new_points[..., 1] = (new_points[..., 1] + 1) * \
            (float(shape[0]-1) / 2.)
    # Convert back to inhomogeneous and hw format
    if points_format == "hw":
        new_points = new_points[..., [1, 0]]
    return new_points


def warp_lines(lines, H):
    """ Warp lines of the shape [N, 2, 2] by an homography H. """
    return warp_points(lines.reshape(-1, 2), H).reshape(-1, 2, 2)

# Warp batch tensor of points using homography
def warp_points_batch(points: torch.Tensor, homography, points_format="hw", pre_normalized=False, shape=None):
    assert points_format in ["xy", "hw"]
    if points_format == "hw":
        new_points = torch.cat([points[..., [1, 0]],
                                torch.ones_like(points[..., :1])], dim=-1)
    else:
        new_points = torch.cat([points,
                                torch.ones_like(points[..., :1])], dim=-1)
    if pre_normalized:
        new_points[..., 0] = new_points[..., 0] / (float(shape[1]-1) / 2.) - 1
        new_points[..., 1] = new_points[..., 1] / (float(shape[0]-1) / 2.) - 1
    new_points = torch.bmm(
        homography, new_points.transpose(1, 2)).transpose(1, 2)
    new_points = new_points[..., :2] / new_points[..., 2:]
    if pre_normalized:
        # Rescale to original size
        new_points[..., 0] = (new_points[..., 0] + 1) * \
            (float(shape[1]-1) / 2.)
        new_points[..., 1] = (new_points[..., 1] + 1) * \
            (float(shape[0]-1) / 2.)
    # Convert back to hw format
    if points_format == "hw":
        new_points = new_points[..., [1, 0]]

    return new_points


# Mask out the points that are outside of img_size
def mask_points(points, img_size):
    mask = ((points[..., 0] >= 0)
            & (points[..., 0] < img_size[0])
            & (points[..., 1] >= 0)
            & (points[..., 1] < img_size[1]))
    return mask


def mask_points_bound(points, img_bound):
    mask = ((points[..., 0] >= img_bound[0])
            & (points[..., 0] < img_bound[2])
            & (points[..., 1] >= img_bound[1])
            & (points[..., 1] < img_bound[3]))
    return mask

# warp points and get mask to filter the point
def mask_warped_points(points, homography, img_size):
    warp_pts = warp_points(points, homography)
    mask = mask_points(warp_pts, img_size)
    return warp_pts[mask], mask


# Convert a tensor [N, 2] or batched tensor [B, N, 2] of N keypoints into
# a grid in [-1, 1]² that can be used in torch.nn.functional.interpolate
def keypoints_to_grid(keypoints, img_size):
    n_points = keypoints.size()[-2]
    device = keypoints.device
    grid_points = keypoints.float() * 2. / torch.tensor(
        img_size, dtype=torch.float, device=device).sub(1) - 1.
    grid_points = grid_points[..., [1, 0]].view(-1, n_points, 1, 2)
    return grid_points


# Return a 2D matrix indicating the local neighborhood of each point
# for a given threshold and two lists of corresponding keypoints
def get_dist_mask(kp0, kp1, valid_mask, dist_thresh):
    b_size, n_points, _ = kp0.size()
    dist_mask0 = torch.norm(kp0.unsqueeze(2) - kp0.unsqueeze(1), dim=-1)
    dist_mask1 = torch.norm(kp1.unsqueeze(2) - kp1.unsqueeze(1), dim=-1)
    dist_mask = torch.min(dist_mask0, dist_mask1)
    dist_mask = dist_mask <= dist_thresh
    dist_mask = dist_mask.repeat(1, 1, b_size).reshape(b_size * n_points,
                                                       b_size * n_points)
    if valid_mask is not None:
        dist_mask = dist_mask[valid_mask, :][:, valid_mask]
    return dist_mask


# Return a 2D matrix indicating for each pair of points
# if they are on the same line or not
def get_common_line_mask(line_indices, valid_mask):
    b_size, n_points = line_indices.shape
    common_mask = line_indices[:, :, None] == line_indices[:, None, :]
    common_mask = common_mask.repeat(1, 1, b_size).reshape(b_size * n_points,
                                                           b_size * n_points)
    common_mask = common_mask[valid_mask, :][:, valid_mask]
    return common_mask


# Return a mask of the valid lines that are within a valid mask of an image
def mask_lines(lines, mask):
    if mask.ndim == 1:
        # Filter out the out-of-border lines if necessary
        boundary_valid = (lines < 0).sum(-1).sum(-1)
        boundary_valid += (lines[..., 0] >= mask[0]).sum(-1)
        boundary_valid += (lines[..., 1] >= mask[1]).sum(-1)
        valid = (boundary_valid == 0)
    else:
        h, w = mask.shape
        int_lines = np.clip(np.round(lines).astype(int), 0, [h - 1, w - 1])
        h_valid = mask[int_lines[:, 0, 0], int_lines[:, 0, 1]].astype(np.bool8)
        w_valid = mask[int_lines[:, 1, 0], int_lines[:, 1, 1]].astype(np.bool8)
        valid = h_valid & w_valid

    return lines[valid, :], valid


def project_point_to_line(line_segs, points):
    """ Given a list of line segments and a list of points (2D or 3D coordinates),
        compute the orthogonal projection of all points on all lines.
        This returns the 1D coordinates of the projection on the line,
        as well as the list of orthogonal distances. """
    # Compute the 1D coordinate of the points projected on the line
    dir_vec = (line_segs[:, 1] - line_segs[:, 0])[:, None]
    coords1d = (((points[None] - line_segs[:, None, 0]) * dir_vec).sum(axis=2)
                / np.linalg.norm(dir_vec, axis=2) ** 2)
    # coords1d is of shape (n_lines, n_points)
    
    # Compute the orthogonal distance of the points to each line
    projection = line_segs[:, None, 0] + coords1d[:, :, None] * dir_vec
    dist_to_line = np.linalg.norm(projection - points[None], axis=2)

    return coords1d, dist_to_line


def get_segment_overlap(seg_coord1d):
    """ Given a list of segments parameterized by the 1D coordinate
        of the endpoints, compute the overlap with the segment [0, 1]. """
    seg_coord1d = np.sort(seg_coord1d, axis=-1)
    overlap = ((seg_coord1d[..., 1] > 0) * (seg_coord1d[..., 0] < 1)
               * (np.minimum(seg_coord1d[..., 1], 1)
                  - np.maximum(seg_coord1d[..., 0], 0)))
    return overlap


def get_overlap_orth_line_dist(line_seg1, line_seg2, min_overlap=0.5,
                               return_overlap=False, mode='min'):
    """ Compute the symmetrical orthogonal line distance between two sets
        of lines and the average overlapping ratio of both lines.
        Enforce a high line distance for small overlaps.
        This is compatible for nD objects (e.g. both lines in 2D or 3D). """
    n_lines1, n_lines2 = len(line_seg1), len(line_seg2)

    # Compute the average orthogonal line distance
    coords_2_on_1, line_dists2 = project_point_to_line(
        line_seg1, line_seg2.reshape(n_lines2 * 2, -1))
    line_dists2 = line_dists2.reshape(n_lines1, n_lines2, 2).sum(axis=2)
    coords_1_on_2, line_dists1 = project_point_to_line(
        line_seg2, line_seg1.reshape(n_lines1 * 2, -1))
    line_dists1 = line_dists1.reshape(n_lines2, n_lines1, 2).sum(axis=2)
    line_dists = (line_dists2 + line_dists1.T) / 2

    # Compute the average overlapping ratio
    coords_2_on_1 = coords_2_on_1.reshape(n_lines1, n_lines2, 2)
    overlaps1 = get_segment_overlap(coords_2_on_1)
    coords_1_on_2 = coords_1_on_2.reshape(n_lines2, n_lines1, 2)
    overlaps2 = get_segment_overlap(coords_1_on_2).T
    overlaps = (overlaps1 + overlaps2) / 2
    min_overlaps = np.minimum(overlaps1, overlaps2)

    if return_overlap:
        return line_dists, overlaps

    # Enforce a max line distance for line segments with small overlap
    if mode == 'mean':
        low_overlaps = overlaps < min_overlap
    else:
        low_overlaps = min_overlaps < min_overlap
    line_dists[low_overlaps] = np.amax(line_dists)
    return line_dists


@lru_cache()
def image_grid(H, W, normalized=False):
    """ Create meshgrid of image
    """
    if normalized:
        xs = torch.linspace(-1, 1, W)
        ys = torch.linspace(-1, 1, H)
    else:
        xs = torch.linspace(0, W-1, W)
        ys = torch.linspace(0, H-1, H)
    ys, xs = torch.meshgrid([ys, xs], indexing="ij")
    coords = [ys, xs]
    grid = torch.stack(coords, dim=0)
    return grid

# Warp img using normalized homography
def warp_img_norm(img, homography, device="cpu"):
    H, W = img.shape[-2::]
    is_numpy = False
    if isinstance(img, np.ndarray):
        is_numpy = True
        img = torch.from_numpy(img).float().unsqueeze(0).to(device)
        homography = torch.from_numpy(
            homography).float().unsqueeze(0).to(device)

    ref_grid = image_grid(H, W, True).to(device).permute(1, 2, 0)
    ref_grid = warp_points_batch(
        ref_grid.unsqueeze(0).reshape(1, -1, 2),
        homography).reshape(-1, H, W, 2).flip(dims=[-1])
    warped_img = F.grid_sample(
        img, ref_grid, mode="bilinear", align_corners=True)

    if is_numpy:
        warped_img = warped_img[0].cpu().numpy()
    return warped_img


def clip_line_to_boundary(lines):
    """ Clip the first coordinate of a set of lines to the lower boundary 0
        and indicate which lines are completely outside of the boundary.
    Args:
        lines: a [N, 2, 2] tensor of lines.
    Returns:
        The clipped coordinates + a mask indicating invalid lines.
    """
    updated_lines = lines.copy()

    # Detect invalid lines completely outside of the first boundary
    invalid = np.all(lines[:, :, 0] < 0, axis=1)

    # Clip the lines to the boundary and update the second coordinate
    # First endpoint
    out = lines[:, 0, 0] < 0
    denom = lines[:, 1, 0] - lines[:, 0, 0]
    denom[denom == 0] = 1e-6
    ratio = lines[:, 1, 0] / denom
    updated_y = ratio * lines[:, 0, 1] + (1 - ratio) * lines[:, 1, 1]
    updated_lines[out, 0, 1] = updated_y[out]
    updated_lines[out, 0, 0] = 0
    # Second endpoint
    out = lines[:, 1, 0] < 0
    denom = lines[:, 0, 0] - lines[:, 1, 0]
    denom[denom == 0] = 1e-6
    ratio = lines[:, 0, 0] / denom
    updated_y = ratio * lines[:, 1, 1] + (1 - ratio) * lines[:, 0, 1]
    updated_lines[out, 1, 1] = updated_y[out]
    updated_lines[out, 1, 0] = 0

    return updated_lines, invalid


def clip_line_to_boundaries(lines, img_size, min_len=10):
    """ Clip a set of lines to the image boundaries and indicate
        which lines are completely outside of the boundaries.
    Args:
        lines: a [N, 2, 2] tensor of lines.
        img_size: the original image size.
    Returns:
        The clipped coordinates + a mask indicating valid lines.
    """
    new_lines = lines.copy()

    # Clip the first coordinate to the 0 boundary of img1
    new_lines, invalid_x0 = clip_line_to_boundary(lines)

    # Mirror in first coordinate to clip to the H-1 boundary
    new_lines[:, :, 0] = img_size[0] - 1 - new_lines[:, :, 0]
    new_lines, invalid_xh = clip_line_to_boundary(new_lines)
    new_lines[:, :, 0] = img_size[0] - 1 - new_lines[:, :, 0]

    # Swap the two coordinates, perform the same for y, and swap back
    new_lines = new_lines[:, :, [1, 0]]
    new_lines, invalid_y0 = clip_line_to_boundary(new_lines)
    new_lines[:, :, 0] = img_size[1] - 1 - new_lines[:, :, 0]
    new_lines, invalid_yw = clip_line_to_boundary(new_lines)
    new_lines[:, :, 0] = img_size[1] - 1 - new_lines[:, :, 0]
    new_lines = new_lines[:, :, [1, 0]]

    # Merge all the invalid lines and also remove lines that became too short
    short = np.linalg.norm(new_lines[:, 1] - new_lines[:, 0],
                           axis=1) < min_len
    valid = np.logical_not(invalid_x0 | invalid_xh
                           | invalid_y0 | invalid_yw | short)

    return new_lines, valid


def overlap_distance_asym(line_seg1, line_seg2):
    """ Compute the overlap distance of line_seg2 projected to line_seg1. """
    n_lines1, n_lines2 = len(line_seg1), len(line_seg2)

    # Project endpoints 2 onto lines 1
    coords_2_on_1, _ = project_point_to_line(
        line_seg1, line_seg2.reshape(n_lines2 * 2, 2))
    coords_2_on_1 = coords_2_on_1.reshape(n_lines1, n_lines2, 2)

    # Compute the overlap
    overlaps = get_segment_overlap(coords_2_on_1)
    return overlaps


def overlap_distance_sym(line_seg1, line_seg2):
    """ Compute the symmetric overlap distance of line_seg2 and line_seg1. """
    overlap_2_on_1 = overlap_distance_asym(line_seg1, line_seg2)
    overlap_1_on_2 = overlap_distance_asym(line_seg2, line_seg1).T
    return (overlap_2_on_1 + overlap_1_on_2) / 2
