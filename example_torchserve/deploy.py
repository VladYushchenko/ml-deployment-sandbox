import os
from argparse import ArgumentParser

import torch
from easyocr.detection import copyStateDict

from example_torchserve.models.QuantizedCRAFT import QuantizedCRAFT

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--detector_path", required=True, type=str, help="Path to the detector"
    )
    parser.add_argument(
        "--output_dir", default="./", type=str, help="Path to the original detector"
    )
    args = parser.parse_args()

    detector_model = QuantizedCRAFT()
    raw_state_dict = torch.load(args.detector_path)
    state_dict = copyStateDict(raw_state_dict)

    detector_model.load_state_dict(state_dict)
    detector_model.eval()

    print("Detector architecture:\n", detector_model)

    quant_types = {
        torch.nn.Conv2d,
        torch.nn.BatchNorm2d,
        torch.nn.ReLU,
        torch.nn.MaxPool2d,
    }
    model = torch.quantization.quantize_dynamic(
        detector_model, qconfig_spec=quant_types, dtype=torch.qint8
    )
    model_jit = torch.jit.script(model)

    os.makedirs(args.output_dir, exist_ok=True)
    save_jit_model_path = os.path.abspath(
        os.path.join(args.output_dir, f"{model.__class__.__name__}.pt")
    )
    torch.jit.save(model_jit, save_jit_model_path)
