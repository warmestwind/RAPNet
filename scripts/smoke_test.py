"""Check RAPNet's tensor interface using random weights and synthetic data.

This is an execution check, not a reproduction of the paper's accuracy.
No patient data, pretrained checkpoint, or CUDA device is required.
"""

import argparse
import json
import platform
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from RAP import RAP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "settings.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.shot, args.batch_size, args.threads) < 1:
        parser.error("shot, batch-size and threads must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    with args.config.open(encoding="utf-8") as handle:
        params = yaml.safe_load(handle)["NETWORK"]
    model = RAP(deepcopy(params)).to(device).eval()
    query = torch.rand(args.batch_size, 3, 256, 256, device=device)
    images = torch.rand(args.shot, args.batch_size, 3, 256, 256, device=device)
    masks = torch.zeros_like(images)
    masks[..., 64:192, 80:176] = 1
    condition = torch.cat((images, masks), dim=2)
    with torch.no_grad():
        outputs = model(query, condition, masks)

    expected = (args.batch_size, 1, 256, 256)
    names = ("probability", "registered_prior", "prototype_mask")
    summary = {}
    for name, output in zip(names, outputs):
        if tuple(output.shape) != expected:
            raise AssertionError(f"{name}: expected {expected}, got {tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise AssertionError(f"{name} contains non-finite values")
        summary[name] = {"shape": list(output.shape), "finite": True}
    if not ((outputs[0] >= 0) & (outputs[0] <= 1)).all():
        raise AssertionError("probability output is outside [0, 1]")
    print(json.dumps({
        "status": "PASS",
        "scope": "synthetic forward pass with random weights; no accuracy claim",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "shot": args.shot,
        "seed": args.seed,
        "outputs": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
