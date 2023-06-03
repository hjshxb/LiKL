import torch
import numpy as np

from likl.models.model_util import get_model
from likl.config.misc import load_config


class LiklDetector(object):
    def __init__(self, model_path, model_cfg_path, pts_cfg, lines_cfg) -> None:
        self.pts_cfg = pts_cfg
        self.lines_cfg = lines_cfg
        model_cfg = load_config(model_cfg_path)
        self.model = get_model(model_cfg["model_cfg"], "test")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def run(self, img, return_type="dict"):
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = img[..., None].repeat(3, 2)
            assert img.dtype == np.float32, 'Image must be float32.'
            inp = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        else:
            if img.ndim == 3:
                img = img.unsqueeze(0)
            if img.shape[1] == 1:
                img = torch.cat([img, img, img], dim=1)
            inp = img

        inp = inp.to(self.device)
        # Extract feature
        output = self.model.inference(
            inp, self.pts_cfg, self.lines_cfg)
        batch_pts = output["batch_pts"]
        batch_pts_desc = output["batch_pts_desc"]
        for i in range(len(batch_pts)):
            # hw format ==> xy format
            batch_pts[i] = batch_pts[i][:, [1, 0, 2]]
            batch_pts_desc[i] = batch_pts_desc[i].cpu().numpy()
        output["batch_pts"] = batch_pts
        output["batch_pts_desc"] = batch_pts_desc

        if return_type == "dict":
            return output
        else:
            raise ValueError("Unsupported return type")
