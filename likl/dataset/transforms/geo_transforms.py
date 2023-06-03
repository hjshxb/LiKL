"""
Code adapted from https://github.com/cvg/SOLD2
"""
import cv2
import numpy as np
import shapely.geometry as sg

from ...misc.geometry_utils import mask_points_bound

__all__ = ["random_scaling", "random_flipup"]


def random_scaling(image, lines, points, scale=1., h_crop=0, w_crop=0):
    """
    return
    ------
    image
    lines (N, 2, (h, w))
    points (N, (h, w))
    valid_mask
    """
    h, w = image.shape[:2]
    h_scale, w_scale = round(h * scale), round(w * scale)

    # Nothing to do if the scale is too close to 1
    if h_scale == h and w_scale == w:
        return (image, lines, points, np.ones([h, w], dtype=np.int32))

    # Zoom-in => resize and random crop
    if scale >= 1.:
        image_big = cv2.resize(image, (w_scale, h_scale),
                               interpolation=cv2.INTER_LINEAR)
        # Crop the image
        image = image_big[h_crop:h_crop+h, w_crop:w_crop+w, ...]
        valid_mask = np.ones([h, w], dtype=np.int32)

        # Process lines
        lines = lines * scale
        lines = _process_lines(
            lines, h_crop, w_crop, h, w, "zoom-in")

        # Process points
        points = points * scale
        img_bound = np.array(
            [h_crop, w_crop, h_crop + h, w_crop + w], np.float32)
        mask = mask_points_bound(points, img_bound)
        points = points[mask]
        points[..., 0] -= h_crop
        points[..., 1] -= w_crop

    # Zoom-out => resize and pad
    else:
        image_shape_raw = image.shape
        image_small = cv2.resize(image, (w_scale, h_scale),
                                 interpolation=cv2.INTER_AREA)
        # Decide the pasting location
        h_start = round((h - h_scale) / 2)
        w_start = round((w - w_scale) / 2)
        # Paste the image to the middle
        image = np.zeros(image_shape_raw, dtype=np.float32)
        image[h_start:h_start+h_scale,
              w_start:w_start+w_scale, ...] = image_small
        valid_mask = np.zeros([h, w], dtype=np.int32)
        valid_mask[h_start:h_start+h_scale, w_start:w_start+h_scale] = 1
        # Process the lines
        lines[..., 0] = (lines[..., 0] * scale) + h_start
        lines[..., 1] = (lines[..., 1] * scale) + w_start
        # Process the points
        points[..., 0] = (points[..., 0] * scale) + h_start
        points[..., 1] = (points[..., 1] * scale) + w_start

    return image, lines, points, valid_mask


def _process_lines(lines, h_start, w_start, h, w, mode):
    if mode == "zoom-in":
        line_segments = lines.reshape(lines.shape[0], 4)
        # crop line seg
        line_segments_new = np.zeros([0, 4])
        image_poly = sg.Polygon(
            [[w_start, h_start],
             [w_start+w, h_start],
             [w_start+w, h_start+h],
             [w_start, h_start+h]
             ])
        for idx in range(line_segments.shape[0]):
            # Get the line segment
            seg_raw = line_segments[idx, :]   # in HW format.
            # Convert to shapely line (flip to xy format)
            seg = sg.LineString([np.flip(seg_raw[:2]),
                                np.flip(seg_raw[2:])])
            # The line segment is just inside the image.
            if seg.intersection(image_poly) == seg:
                line_segments_new = np.concatenate(
                    (line_segments_new, seg_raw[None, ...]), axis=0)
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
                line_segments_new = np.concatenate(
                    (line_segments_new, segment), axis=0)
            else:
                continue
        line_segments_new = (np.round(line_segments_new)).astype(np.int)
        # Filter segments with 0 length
        segment_lens = np.linalg.norm(
            line_segments_new[:, :2] - line_segments_new[:, 2:], axis=-1)
        seg_mask = segment_lens != 0
        line_segments_new = line_segments_new[seg_mask, :]
        # reshape
        line_segments_new = line_segments_new.reshape(-1, 2, 2)
        line_segments_new[..., 1] -= w_start
        line_segments_new[..., 0] -= h_start
    return line_segments_new


def get_rotate_flip_cfg():
    output = {
        "rotate": np.random.random() < 0.5,
        "ud_flip": np.random.random() < 0.5,
        "lr_flip": np.random.random() > 0.5}
    return output


def random_flip(image, lines, points, cfg=None):
    """ Randomly flip the image
    """
    if cfg is None:
        cfg = get_rotate_flip_cfg()
    if cfg["ud_flip"]:
        h, w = image.shape[:2]
        image = np.flip(image, 0)
        lines = np.concatenate(
            (h - lines[:, :, 0, None], lines[:, :, 1, None]), axis=-1)
        points = np.concatenate(
            (h - points[:, 0, None], points[:, 1, None]), axis=-1)
    if cfg["lr_flip"]:
        h, w = image.shape[:2]
        image = np.flip(image, 1)
        lines = np.concatenate(
            (lines[:, :, 0, None], w - lines[:, :, 1, None]), axis=-1)
        points = np.concatenate(
            (points[:, 0, None], w - points[:, 1, None]), axis=-1)
    return image, lines, points


def random_rotate(image, lines, points, cfg=None):
    """ Randomly rotate the image
    """
    if cfg is None:
        cfg = get_rotate_flip_cfg()
    if cfg["rotate"]:
        h, w = image.shape[:2]
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        lines = np.concatenate(
            (lines[:, :, 1, None], h - lines[:, :, 0, None]), axis=-1)
        points = np.concatenate(
            (points[:, 1, None], h - points[:, 0, None]), axis=-1)
    return image, lines, points


def random_gray(image):
    """ Randomly convert image to gray
    """
    if np.random.random() < 0.5:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image
