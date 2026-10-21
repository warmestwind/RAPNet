"""Create synthetic NRRD inputs and random weights for an I/O demonstration.

The checkpoint is untrained and must not be used to assess segmentation quality.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nrrd
import numpy as np
import torch
import yaml
from RAP import RAP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be new or empty")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    with (ROOT / "settings.yaml").open(encoding="utf-8") as handle:
        model = RAP(deepcopy(yaml.safe_load(handle)["NETWORK"]))
    for directory in ("query", "support"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)
    geometry = {"space": "left-posterior-superior", "space directions": np.diag([0.7, 0.8, 1.0]),
                "space origin": np.array([10.0, 20.0, 30.0])}
    image = rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)
    mask = np.zeros_like(image)
    mask[8:24, 10:30, :] = 1
    nrrd.write(str(args.output_dir / "query" / "synthetic_query_im.nrrd"), image, header=geometry)
    for i in range(2):
        support = rng.integers(0, 256, image.shape, dtype=np.uint8)
        nrrd.write(str(args.output_dir / "support" / f"synthetic_support_{i}_im.nrrd"), support, header=geometry)
        nrrd.write(str(args.output_dir / "support" / f"synthetic_support_{i}_m.nrrd"), mask, header=geometry)
    torch.save({"state_dict": model.state_dict(), "note": "RANDOM UNTRAINED WEIGHTS; I/O CHECK ONLY"},
               args.output_dir / "random_weights.pth")
    (args.output_dir / "README.txt").write_text(
        "All images and masks are synthetic. random_weights.pth is untrained.\n"
        "Use --shot 1 or --shot 2 for this demo. No accuracy claim is possible.\n", encoding="utf-8")
    print(f"Synthetic demo created at {args.output_dir}")


if __name__ == "__main__":
    main()
