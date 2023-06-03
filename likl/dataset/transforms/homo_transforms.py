"""
This file implements the homographic transforms for data augmentation.
Code adapted from https://github.com/cvg/SOLD2
"""


import cv2
import numpy as np
from math import pi
import shapely.geometry

from ...misc.geometry_utils import (warp_points,
                                    mask_points)


def sample_homography(
        shape, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=100, n_angles=100, scaling_amplitude=0.2, perspective_amplitude=0.4,
        patch_ratio=0.7, max_angle=pi/4, inverse=False):
    """ Sample a random homography that includes perspective, scale, translation and rotation operations."""

    width = float(shape[1])
    hw_ratio = float(shape[0]) / float(shape[1])

    pts1 = np.stack([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], axis=0)
    pts2 = pts1.copy() * patch_ratio
    pts2[:, 1] *= hw_ratio

    if perspective:

        perspective_amplitude_x = np.random.normal(
            0., perspective_amplitude/2, (2))
        perspective_amplitude_y = np.random.normal(
            0., hw_ratio * perspective_amplitude/2, (2))

        perspective_amplitude_x = np.clip(
            perspective_amplitude_x, -perspective_amplitude/2, perspective_amplitude/2)
        perspective_amplitude_y = np.clip(
            perspective_amplitude_y, hw_ratio * -perspective_amplitude/2, hw_ratio * perspective_amplitude/2)

        pts2[0, 0] -= perspective_amplitude_x[1]
        pts2[0, 1] -= perspective_amplitude_y[1]

        pts2[1, 0] -= perspective_amplitude_x[0]
        pts2[1, 1] += perspective_amplitude_y[1]

        pts2[2, 0] += perspective_amplitude_x[1]
        pts2[2, 1] -= perspective_amplitude_y[0]

        pts2[3, 0] += perspective_amplitude_x[0]
        pts2[3, 1] += perspective_amplitude_y[0]

    if scaling:

        random_scales = np.random.normal(1, scaling_amplitude/2, (n_scales))
        random_scales = np.clip(
            random_scales, 1-scaling_amplitude/2, 1+scaling_amplitude/2)

        scales = np.concatenate([[1.], random_scales], 0)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = np.expand_dims(pts2 - center, axis=0) * np.expand_dims(
            np.expand_dims(scales, 1), 1) + center
        valid = np.arange(n_scales)  # all scales are valid except scale=1
        idx = valid[np.random.randint(valid.shape[0])]
        pts2 = scaled[idx]

    if translation:
        t_min, t_max = np.min(
            pts2 - [-1., -hw_ratio], axis=0), np.min([1., hw_ratio] - pts2, axis=0)
        pts2 += np.expand_dims(np.stack([np.random.uniform(-t_min[0], t_max[0]),
                                         np.random.uniform(-t_min[1], t_max[1])]),
                               axis=0)

    if rotation:
        angles = np.linspace(-max_angle, max_angle, n_angles)
        angles = np.concatenate([[0.], angles], axis=0)

        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul(
            np.tile(np.expand_dims(pts2 - center, axis=0), [n_angles+1, 1, 1]),
            rot_mat) + center

        valid = np.where(np.all((rotated >= [-1., -hw_ratio]) & (rotated < [1., hw_ratio]),
                                axis=(1, 2)))[0]
        if valid.shape[0] == 0:
            return sample_homography(
                shape, perspective, scaling, rotation, translation,
                n_scales, n_angles, scaling_amplitude,
                perspective_amplitude,
                patch_ratio, max_angle)
        idx = valid[np.random.randint(valid.shape[0])]
        pts2 = rotated[idx]

    pts2[:, 1] /= hw_ratio

    # Rescale to actual size
    if not isinstance(shape, np.ndarray):
        shape = np.array(shape)
    shape = shape[::-1].astype(np.float32)  # different convention [y, x]
    pts1 = (pts1 + 1) * (shape[None, ...] / 2.)
    pts2 = (pts2 + 1) * (shape[None, ...] / 2.)


    # def ax(p, q): return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]
    # def ay(p, q): return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    # a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4)
    #                  for f in (ax, ay)], axis=0)
    # p_mat = np.transpose(np.stack(
    #     [[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))

    # homography = np.matmul(np.linalg.pinv(a_mat), p_mat).squeeze()
    # homography = np.concatenate([homography, [1.]]).reshape(3, 3)
    homography = cv2.getPerspectiveTransform(pts1.astype(np.float32), pts2.astype(np.float32))

    if inverse:
        homography = np.linalg.inv(homography)
    return homography


def compute_valid_mask(image_size, homography,
                       border_margin, valid_mask=None):
    # Warp the mask
    if valid_mask is None:
        initial_mask = np.ones(image_size)
    else:
        initial_mask = valid_mask
    mask = cv2.warpPerspective(
        initial_mask, homography, (image_size[1], image_size[0]),
        flags=cv2.INTER_NEAREST)
    # Optionally perform erosion
    if border_margin > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (border_margin*2, )*2)
        mask = cv2.erode(mask, kernel)

    # Perform dilation if border_margin is negative
    if border_margin < 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (abs(int(border_margin))*2, )*2)
        mask = cv2.dilate(mask, kernel)

    return mask


def warp_line_segment(line_segments, homography, image_size):
    """ Warp the line segments using a homography. """
    # Separate the line segements into 2N points to apply matrix operation
    num_segments = line_segments.shape[0]
    line_segments = line_segments.reshape(num_segments, 4)
    junctions = np.concatenate(
        (line_segments[:, :2],  # The first junction of each segment.
         line_segments[:, 2:]),  # The second junction of each segment.
        axis=0)
    # Convert to homogeneous coordinates
    warped_junctions = warp_points(junctions, homography)
    warped_segments = np.concatenate(
        (warped_junctions[:num_segments, :],
         warped_junctions[num_segments:, :]),
        axis=1
    )

    # Check the intersections with the boundary
    warped_segments_new = np.zeros([0, 4])
    image_poly = shapely.geometry.Polygon(
        [[0, 0], [image_size[1]-1, 0], [image_size[1]-1, image_size[0]-1],
         [0, image_size[0]-1]])
    for idx in range(warped_segments.shape[0]):
        # Get the line segment
        seg_raw = warped_segments[idx, :]   # in HW format.
        # Convert to shapely line (flip to xy format)
        seg = shapely.geometry.LineString([np.flip(seg_raw[:2]),
                                           np.flip(seg_raw[2:])])

        # The line segment is just inside the image.
        if seg.intersection(image_poly) == seg:
            warped_segments_new = np.concatenate((warped_segments_new,
                                                  seg_raw[None, ...]), axis=0)

        # Intersect with the image.
        elif seg.intersects(image_poly):
            # Check intersection
            try:
                p = np.array(
                    seg.intersection(image_poly).coords).reshape([-1, 4])
            # If intersect at exact one point, just continue.
            except:
                continue
            segment = np.concatenate([np.flip(p[0, :2]), np.flip(p[0, 2:],
                                     axis=0)])[None, ...]
            warped_segments_new = np.concatenate(
                (warped_segments_new, segment), axis=0)

        else:
            continue

    warped_segments = (np.round(warped_segments_new)).astype(np.int32)
    warped_segments = warped_segments.reshape(-1, 2, 2)
    return warped_segments


class homography_transform(object):
    """ # Homography transformations. """

    def __init__(self, image_size, homograpy_config,
                 border_margin=0, min_label_len=20):
        self.homo_config = homograpy_config
        self.image_size = image_size
        self.target_size = (self.image_size[1], self.image_size[0])
        self.border_margin = border_margin
        if (min_label_len < 1) and isinstance(min_label_len, float):
            raise ValueError("min_label_len should be in pixels.")
        self.min_label_len = min_label_len

    def __call__(self, input_image, line_segments, points, valid_mask=None,
                 homo=None):
        # Sample one random homography or use the given one
        if homo is None:
            homo = sample_homography(self.image_size,
                                     **self.homo_config)
            
        valid_mask = compute_valid_mask(self.image_size, homo,
                                        self.border_margin, valid_mask)
        
        if (homo == np.eye(3, dtype=homo.dtype)).all():
            return {
                "lines": line_segments,
                "points": points,
                "warped_image": input_image,
                "valid_mask": valid_mask,
                "homo": homo
            }

        # Warp the image
        warped_image = cv2.warpPerspective(
            input_image, homo, self.target_size, flags=cv2.INTER_LINEAR)


        # Warp the segments and check the length.
        # Adjust the min_label_length
        warped_segments = warp_line_segment(
            line_segments, homo, self.image_size)

        # Warp the points
        warped_points = warp_points(points, homo)
        mask = mask_points(warped_points, self.image_size)
        warped_points = warped_points[mask]

        return {
            "lines": warped_segments,
            "points": warped_points,
            "warped_image": warped_image,
            "valid_mask": valid_mask,
            "homo": homo
        }
