"""
Convert the PyTorch model to other formats.
"""

import torch
import argparse

from likl.models.model_util import get_model
from likl.config.misc import load_config


def convert_torchscript(model, img, file):
    prefix = "torchscript"
    """ TorchScript model convert
    """
    print(
        f"[Info]: Starting convert to {prefix} with torch {torch.__version__}")
    ts = torch.jit.trace(model, img, strict=False)
    print(ts.code)
    print(ts.graph)
    print(f"[Info]: {prefix} convert success. Save as {file}")
    ts.save(file)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str,
                        default='likl/config/train_full_cfg.yaml', help='Path to model config')
    parser.add_argument('--weights', type=str, help='Path to model.pt')
    parser.add_argument('--img_size', nargs='+', type=int,
                        default=[480, 480], help='image (h, w)')
    parser.add_argument('--device', default='cuda', help='cuda or cpu device')
    parser.add_argument('--fmts', nargs='+',
                        default=['torchscript'], help='torchscript, onnx')
    parser.add_argument('--file', type=str, default="Path to convert file")
    args = parser.parse_args()
    return args


def main(cfg_path, weights, img_size, device, fmts, file):
    fmts = [f.lower() for f in fmts]
    check = ['torchscript', 'onnx']
    for x in fmts:
        assert x in check, f"Error: Invaild {x}. Valid {fmts}"

    # Load pytorch model
    model_cfg = load_config(cfg_path)["model_cfg"]
    model = get_model(model_cfg, "test")

    state_dict= torch.load(weights, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # im
    img = torch.zeros(1, 3, *img_size).to(device)
    print("[Info]: img size ", img_size)
    if "torchscript" in fmts:
        convert_torchscript(model, img, file)
    if "onnx" in fmts:
        raise NotImplementedError


if __name__ == "__main__":
    args = parser_args()
    main(**vars(args))
