"""Compatibility entry point for the safe NRRD inference command.

Run ``python test.py --help`` or prefer ``python inference.py --help``.
The previous hard-coded paths and automatic output-directory deletion are removed.
"""

from pathlib import Path
from types import SimpleNamespace

from inference import main, run


def ts_main(ckpt_path, *, query_dir=None, support_dir=None, output_dir=None,
            config=None, device="cpu", shot=5, target_label=1, seed=0, threads=2):
    """Run inference with explicit paths; no query annotation is required."""
    if query_dir is None or support_dir is None or output_dir is None:
        raise ValueError("Provide query_dir, support_dir and output_dir explicitly; see python test.py --help")
    return run(SimpleNamespace(
        checkpoint=Path(ckpt_path), query_dir=Path(query_dir),
        support_dir=Path(support_dir), output_dir=Path(output_dir),
        config=Path(config) if config else Path(__file__).resolve().with_name("settings.yaml"),
        device=device, shot=shot, target_label=target_label, seed=seed, threads=threads,
    ))


if __name__ == "__main__":
    main()
