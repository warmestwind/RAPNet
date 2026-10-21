"""Run RAPNet on NRRD query volumes using labelled NRRD support volumes."""

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import cv2
import nrrd
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from RAP import RAP

GEOMETRY_KEYS = (
    "space", "space dimension", "space directions", "space origin",
    "spacings", "kinds", "centers", "labels", "units", "space units",
    "measurement frame", "thicknesses",
)


def read_volume(path):
    """Match the original [X,Y,S] NRRD -> [S,Y,X] array convention."""
    array, header = nrrd.read(str(path))
    if array.ndim != 3 or min(array.shape) < 1:
        raise ValueError(f"Expected a nonempty 3-D NRRD volume: {path}")
    if array.dtype.kind not in "uif" or not np.isfinite(array).all():
        raise ValueError(f"Volume must contain finite numeric values: {path}")
    return array.transpose(2, 1, 0), header


def triplet(volume, center, mask=False, target_label=1):
    indices = [max(0, center - 1), center, min(len(volume) - 1, center + 1)]
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR
    frames = []
    for i in indices:
        frame = volume[i]
        # Keep the release's integer-image resize rounding where OpenCV supports it.
        if frame.dtype not in (np.uint8, np.uint16, np.int16, np.float32, np.float64):
            frame = frame.astype(np.float32)
        frames.append(cv2.resize(frame, (256, 256), interpolation=interpolation))
    result = np.stack(frames).astype(np.float32)
    result = (result == target_label).astype(np.float32) if mask else result / 255.0
    return torch.from_numpy(np.ascontiguousarray(result))


def check_support_geometry(image_header, mask_header, image_path, mask_path):
    """Reject conflicting shared spatial metadata; never resample labels silently."""
    keys = ("space", "space dimension", "space directions", "space origin",
            "spacings", "space units", "measurement frame")
    for key in keys:
        if key not in image_header or key not in mask_header:
            continue
        left, right = np.asarray(image_header[key]), np.asarray(mask_header[key])
        equal = left.shape == right.shape
        if equal and left.dtype.kind in "buif" and right.dtype.kind in "buif":
            equal = np.allclose(left, right, rtol=1e-5, atol=1e-6, equal_nan=True)
        elif equal:
            equal = np.array_equal(left, right)
        if not equal:
            raise ValueError(f"Support image/mask geometry mismatch ({key}): "
                             f"{image_path} versus {mask_path}")


def load_model(checkpoint_path, config_path, device):
    with Path(config_path).open(encoding="utf-8") as handle:
        params = yaml.safe_load(handle)["NETWORK"]
    model = RAP(deepcopy(params)).to(device)
    # Accept state_dict checkpoints; do not unpickle arbitrary saved model objects.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state_dict or a dict containing 'state_dict'")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def run(args):
    if args.shot < 1 or args.target_label < 1 or args.threads < 1:
        raise ValueError("shot, target-label and threads must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; pass --device cpu")
    query_dir = Path(args.query_dir)
    support_dir = Path(args.support_dir)
    output_dir = Path(args.output_dir)
    if not query_dir.is_dir() or not support_dir.is_dir():
        raise ValueError("query-dir and support-dir must be existing directories")
    queries = sorted(query_dir.glob("*_im.nrrd"))
    support_paths = sorted(support_dir.glob("*_im.nrrd"))
    if not queries:
        raise ValueError(f"No *_im.nrrd query volumes in {query_dir}")
    if not support_paths:
        raise ValueError(f"No *_im.nrrd support volumes in {support_dir}")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("output-dir must be new or empty; existing output is never deleted")

    supports = {}
    for path in support_paths:
        mask_path = path.with_name(path.name.removesuffix("_im.nrrd") + "_m.nrrd")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing support mask: {mask_path}")
        image, image_header = read_volume(path)
        mask, mask_header = read_volume(mask_path)
        if image.shape != mask.shape:
            raise ValueError(f"Support image/mask shape mismatch: {path.name}")
        check_support_geometry(image_header, mask_header, path, mask_path)
        if not np.equal(mask, np.floor(mask)).all() or np.any(mask < 0):
            raise ValueError(f"Support mask must contain nonnegative integer labels: {mask_path}")
        if not np.any(mask == args.target_label):
            raise ValueError(f"Support mask has no target label {args.target_label}: {mask_path}")
        supports[path] = (image, mask)
    for path in queries:
        available = [s for s in support_paths if s.resolve() != path.resolve()]
        if len(available) < args.shot:
            raise ValueError(f"{path.name}: need {args.shot} distinct support volumes excluding the query")

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = random.Random(args.seed)
    model = load_model(args.checkpoint, args.config, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(Path(args.checkpoint)), "seed": args.seed,
        "shot": args.shot, "target_label": args.target_label,
        "device": str(device), "torch": torch.__version__, "queries": [],
        "note": "Binary research predictions; execution does not establish segmentation accuracy.",
    }
    with torch.no_grad():
        for query_path in queries:
            volume, header = read_volume(query_path)
            candidates = [p for p in support_paths if p.resolve() != query_path.resolve()]
            prediction, prior = [], []
            query_record = {"file": query_path.name, "supports_per_frame": []}
            for frame in range(len(volume)):
                selected = rng.sample(candidates, args.shot)
                images, masks = [], []
                for support_path in selected:
                    support_image, support_mask = supports[support_path]
                    # Preserve the original relative frame-position matching.
                    center = int(frame / len(volume) * len(support_image))
                    mask = triplet(support_mask, center, True, args.target_label)
                    if float(mask[1].sum()) == 256 * 256:
                        raise ValueError(
                            f"{support_path.name}, frame {center}: the resized center mask "
                            "has no background for prototype construction"
                        )
                    images.append(triplet(support_image, center))
                    masks.append(mask)
                query = triplet(volume, frame).unsqueeze(0).to(device)
                support_images = torch.stack(images).unsqueeze(1).to(device)
                support_masks = torch.stack(masks).unsqueeze(1).to(device)
                condition = torch.cat((support_images, support_masks), dim=2)
                probability, registered_prior, _ = model(query, condition, support_masks)
                if not torch.isfinite(probability).all() or not torch.isfinite(registered_prior).all():
                    raise ValueError(f"Non-finite model output for {query_path.name}, frame {frame}")
                size = volume.shape[1:]
                probability = F.interpolate(probability, size=size, mode="bilinear", align_corners=True)
                registered_prior = F.interpolate(registered_prior, size=size, mode="bilinear", align_corners=True)
                prediction.append((probability[0, 0].cpu().numpy() > 0.5).astype(np.uint8))
                prior.append((registered_prior[0, 0].cpu().numpy() > 0.5).astype(np.uint8))
                query_record["supports_per_frame"].append([p.name for p in selected])
            stem = query_path.name.removesuffix("_im.nrrd")
            geometry = {key: deepcopy(header[key]) for key in GEOMETRY_KEYS if key in header}
            for suffix, frames in (("pred", prediction), ("prior", prior)):
                nrrd.write(str(output_dir / f"{stem}_{suffix}.nrrd"),
                           np.stack(frames).transpose(2, 1, 0), header=geometry)
            manifest["queries"].append(query_record)
            print(f"Saved {stem}: {len(volume)} frames at the original array shape")
    (output_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--query-dir", required=True, type=Path)
    parser.add_argument("--support-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().with_name("settings.yaml"))
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0, etc.")
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--target-label", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
