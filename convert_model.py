"""
Convert the PyTorch model to other formats.
"""

import torch
import onnx
import argparse

from likl.models.model_util import get_model
from likl.config.misc import load_config


def convert_torchscript(model, img, file):
    """ TorchScript model convert
    """
    prefix = "torchscript"
    print(
        f"[Info]: Starting convert to {prefix} with torch {torch.__version__}")
    ts = torch.jit.trace(model, img, strict=False)
    print(ts.code)
    print(ts.graph)
    print(f"[Info]: {prefix} convert success. Save as {file}")
    ts.save(file)


def convert_onnx(model, img, file: str):
    """ Exports model to ONNX format
    """
    prefix = "ONNX"
    print(
        f"[Info]: Starting convert to {prefix} with torch {torch.__version__} and onnx {onnx.__version__}")
    # Export model
    import deform_conv2d_onnx_exporter
    deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()
    torch.onnx.export(model, 
                      img, 
                      file, 
                      verbose=False,
                      opset_version=12,
                      input_names=['input'],
                      output_names=['line_pred', 'points_pred', 'desc_pred'],
                      dynamic_axes={'input': {2: 'image_height', 3: "image_width"}})

    # Check
    onnx_model = onnx.load(file)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

    print(f"[Info]: {prefix} convert success. Save as {file}")


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str,
                        default='likl/config/train_full_cfg.yaml', help='Path to model config')
    parser.add_argument('--weights', type=str, help='Path to model.pt')
    parser.add_argument('--img_size', nargs='+', type=int,
                        default=[480, 480], help='image (h, w)')
    parser.add_argument('--device', default='cuda', help='cuda or cpu device')
    parser.add_argument('--fmt', default='torchscript', help='torchscript, onnx')
    parser.add_argument('--file', type=str, default="Path to convert file")
    # parser.add_argument('--simplify', default=False, action="store_true", help="ONNX: simplify model")
    args = parser.parse_args()
    return args


def main(cfg_path, weights, img_size, device, fmt, file):
    fmt = fmt.lower()
    check = ['torchscript', 'onnx']
    assert fmt in check, f"Error: Invaild {fmt}. Valid {check}"

    # Load pytorch model
    model_cfg = load_config(cfg_path)["model_cfg"]
    model = get_model(model_cfg, "test")

    state_dict= torch.load(weights, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # im
    img = torch.zeros(1, 3, *img_size).to(device)
    print("[Info]: img size ", img_size)
    if "torchscript" in fmt:
        convert_torchscript(model, img, file)
    elif "onnx" in fmt:
        convert_onnx(model, img, file)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = parser_args()
    main(**vars(args))
