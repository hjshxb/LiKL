"""
Code from https://github.com/cvg/SOLD2
"""
import cv2
import torch
import numpy as np


def get_line_matcher(matcher_type, cross_check=True):
    if matcher_type == "nn":
        return NNLineMatcher(cross_check)
    elif matcher_type == "binary":
        return BinaryLineMatcher(cross_check)
    elif matcher_type == "wunsch":
        return WunschLineMatcher(cross_check)
    else:
        raise ValueError("Unknown matcher type")


class NNLineMatcher():
    def __init__(self, cross_check=True) -> None:
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)

    def match(self, desc1, desc2):
        bfmatches = self.matcher.match(desc1, desc2)
        query_idx = np.array([m.queryIdx for m in bfmatches])
        train_idx = np.array([m.trainIdx for m in bfmatches])
        matches = -np.ones(len(desc1), dtype=np.int32)
        if len(query_idx) > 0:
            matches[query_idx] = train_idx
        return matches

    def get_pairwise_distance(self, desc1, desc2):
        return np.linalg.norm(desc1 - desc2, axis=1)


class BinaryLineMatcher():
    def __init__(self, cross_check=True) -> None:
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)

    def match(self, desc1, desc2):
        bfmatches = self.matcher.match(desc1, desc2)
        query_idx = np.array([m.queryIdx for m in bfmatches])
        train_idx = np.array([m.trainIdx for m in bfmatches])
        matches = -np.ones(len(desc1), dtype=np.int32)
        if len(query_idx) > 0:
            matches[query_idx] = train_idx
        return matches

    def get_pairwise_distance(self, desc1, desc2):
        return np.array(
            [cv2.norm(desc1[i], desc2[i], cv2.NORM_HAMMING)
             for i in range(len(desc1))])


class WunschLineMatcher(object):
    """ Class matching two sets of line segments
        with the Needleman-Wunsch algorithm. 
        """

    def __init__(self, cross_check=True, top_k_candidates=10, gap=0.1) -> None:
        self.cross_check = cross_check
        self.top_k_candidates = top_k_candidates
        self.gap = gap

    def match(self, desc1, desc2, valid_points1, valid_points2):
        """
        desc1: (len(lines1), N, D)
        desc2: (len(lines2), N, D)
        valid_points1: (len(lines1), N)
        valid_points2: (len(lines2), N)
        """
        assert desc1.shape[1:] == desc2.shape[1:]
        if isinstance(desc1, np.ndarray):
            desc1 = torch.from_numpy(desc1)
            desc2 = torch.from_numpy(desc2)
            valid_points1 = torch.from_numpy(valid_points1)
            valid_points2 = torch.from_numpy(valid_points2)

        num_lines1, num_samples, D = desc1.shape
        num_lines2 = len(desc2)
        desc1 = desc1.reshape(-1, D)
        desc2 = desc2.reshape(-1, D)

        # Cosine distance (num_lines1 * num_samples, num_lines2 * num_samples)
        scores = desc1 @ desc2.t()
        # Assign a score of -1 for unvalid points
        scores[~valid_points1.flatten()] = -1
        scores[:, ~valid_points2.flatten()] = -1
        # (num_lines1, num_lines2, num_samples, num_samples)
        scores = scores.reshape(num_lines1, num_samples,
                                num_lines2, num_samples)
        scores = scores.permute(0, 2, 1, 3)

        matches = self.filter_and_match_lines(scores)

        if self.cross_check:
            matches2 = self.filter_and_match_lines(
                scores.permute(1, 0, 3, 2))
            mutual = matches2[matches] == np.arange(num_lines1)
            matches[~mutual] = -1
        return matches

    @staticmethod
    def sample_line_points(line_seg, num_samples, sample_min_dist):
        """
        Regularly sample points along each line segments, with a minimal
        distance between each point. Pad the remaining points.
        Inputs:
            line_seg: an Nx2x2 np.array.
        Outputs:
            line_points: an Nxnum_samplesx2 np.array. Return the same format (hw or xy) according to the input.
            valid_points: a boolean Nxnum_samples np.array.
        """
        num_lines = len(line_seg)
        line_lengths = np.linalg.norm(line_seg[:, 0] - line_seg[:, 1], axis=1)

        # Sample the points separated by at least min_dist_pts along each line
        # The number of samples depends on the length of the line
        num_samples_lst = np.clip(line_lengths // sample_min_dist,
                                  2, num_samples)
        line_points = np.empty((num_lines, num_samples, 2), dtype=float)
        valid_points = np.empty((num_lines, num_samples), dtype=bool)
        for n in np.arange(2, num_samples + 1):
            # Consider all lines where we can fit up to n points
            cur_mask = num_samples_lst == n
            cur_line_seg = line_seg[cur_mask]
            line_points_x = np.linspace(cur_line_seg[:, 0, 0],
                                        cur_line_seg[:, 1, 0],
                                        n, axis=-1)
            line_points_y = np.linspace(cur_line_seg[:, 0, 1],
                                        cur_line_seg[:, 1, 1],
                                        n, axis=-1)
            cur_line_points = np.stack([line_points_x, line_points_y], axis=-1)

            # Pad
            cur_num_lines = len(cur_line_seg)
            cur_valid_points = np.ones((cur_num_lines, num_samples),
                                       dtype=bool)
            cur_valid_points[:, n:] = False
            cur_line_points = np.concatenate([
                cur_line_points,
                np.zeros((cur_num_lines, num_samples - n, 2), dtype=float)],
                axis=1)

            line_points[cur_mask] = cur_line_points
            valid_points[cur_mask] = cur_valid_points

        return line_points, valid_points

    def filter_and_match_lines(self, scores):
        """
        Use the scores to keep the top k best lines, compute the Needleman-
        Wunsch algorithm on each candidate pairs, and keep the highest score.
        Inputs:
            scores: a (N, M, n, n) torch.Tensor containing the pairwise scores
                    of the elements to match.
        Outputs:
            matches: a (N) np.array containing the indices of the best match
        """
        # Pre-filter the pairs and keep the top k best candidate lines
        line_scores1 = scores.max(3)[0]
        valid_scores1 = line_scores1 != -1
        line_scores1 = ((line_scores1 * valid_scores1).sum(2)
                        / valid_scores1.sum(2))
        line_scores2 = scores.max(2)[0]
        valid_scores2 = line_scores2 != -1
        line_scores2 = ((line_scores2 * valid_scores2).sum(2)
                        / valid_scores2.sum(2))
        line_scores = (line_scores1 + line_scores2) / 2
        # (num_lines1, top_k_candidates)
        topk_lines = torch.argsort(line_scores,
                                   dim=1)[:, -self.top_k_candidates:]
        top_scores = torch.take_along_dim(
            scores, topk_lines[:, :, None, None], dim=1)
        # Consider the reversed line segments as well
        topk_lines = topk_lines.cpu().numpy()
        top_scores = top_scores.cpu().numpy()
        top_scores = np.concatenate([top_scores, top_scores[..., ::-1]],
                                    axis=1)

        # Compute the line distance matrix with Needleman-Wunsch algo and
        # retrieve the closest line neighbor
        n_lines1, top2k, n, m = top_scores.shape
        top_scores = top_scores.reshape(n_lines1 * top2k, n, m)
        nw_scores = self.needleman_wunsch(top_scores)
        nw_scores = nw_scores.reshape(n_lines1, top2k)
        matches = np.mod(np.argmax(nw_scores, axis=1), top2k // 2)
        matches = topk_lines[np.arange(n_lines1), matches]
        return matches

    def needleman_wunsch(self, scores):
        """
        Batched implementation of the Needleman-Wunsch algorithm.
        The cost of the InDel operation is set to 0 by subtracting the gap
        penalty to the scores.
        Inputs:
            scores: a (B, N, M) np.array containing the pairwise scores
                    of the elements to match.
        """
        if scores.ndim == 2:
            scores = scores[np.newaxis, :, :]

        b, n, m = scores.shape
        nw_scores = scores - self.gap
        # Run the dynamic programming algorithm
        nw_grid = np.zeros((b, n + 1, m + 1), dtype=float)
        for i in range(n):
            for j in range(m):
                nw_grid[:, i + 1, j + 1] = np.maximum(
                    np.maximum(nw_grid[:, i + 1, j], nw_grid[:, i, j + 1]),
                    nw_grid[:, i, j] + nw_scores[:, i, j])

        return nw_grid[:, -1, -1]


if __name__ == "__main__":
    # Test the Needleman-Wunsch algorithm
    nw_algo = WunschLineMatcher()
    scores1 = np.eye(4)  # nw_score should be 4 * 0.9 = 3.6
    scores2 = np.array([[1, 0, 0, 0],  # nw_score should be 3 * 0.9 = 2.7
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])
    scores = [scores1, scores2]
    for s in scores:
        print("NW output for an input of:")
        print(s)
        print(nw_algo.needleman_wunsch(s))
        print()
