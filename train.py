"""Configurable entry point for the released CUDA training loop."""

import argparse
import copy

import torch

from utils import config


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def get_configs(argv=None):
    parser = argparse.ArgumentParser(description="RAPNet ultrasound training (CUDA required)")
    parser.add_argument("--config", default="./settings.yaml", help="YAML configuration file")
    parser.add_argument("--mode", "-m", default="train", choices=["train"],
                        help="This entry point implements training only")
    parser.add_argument("--device", "-d", default=None,
                        help="CUDA index or cuda:N; overrides COMMON.device")
    parser.add_argument("--shot", type=_positive_int, default=None,
                        help="Support examples per episode; overrides TRAINING.shot")
    parser.add_argument("--batch-size", type=_positive_int, default=None,
                        help="Overrides TRAINING.train_batch_size")
    parser.add_argument("--num-workers", type=_nonnegative_int, default=None,
                        help="Overrides TRAINING.num_workers")
    args = parser.parse_args(argv)
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg, args


def _cuda_index(value):
    if isinstance(value, int) or str(value).isdigit():
        index = int(value)
    else:
        device = torch.device(value)
        if device.type != "cuda":
            raise ValueError("The released training loop requires CUDA; use the synthetic smoke example for CPU checks.")
        index = 0 if device.index is None else device.index
    if index < 0:
        raise ValueError("CUDA device index must be non-negative.")
    return index


def train(train_params, common_params, data_params, net_params, args=None):
    # Import the full training stack only when training is requested.
    import RAP as fs
    from dataset import transform, dataset_us
    from solver_us_pro import Solver

    device_value = args.device if args is not None and args.device is not None else common_params.get("device", "cuda:0")
    device = _cuda_index(device_value)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build and select an available GPU.")
    if device >= torch.cuda.device_count():
        raise ValueError("CUDA device {} is unavailable; this process can see {} device(s).".format(
            device, torch.cuda.device_count()))
    # Respect CUDA_VISIBLE_DEVICES supplied by the caller; do not overwrite it.
    torch.cuda.set_device(device)

    def setting(flag_name, config_name, default, allow_zero=False):
        override = getattr(args, flag_name, None) if args is not None else None
        value = train_params.get(config_name, default) if override is None else override
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if allow_zero else 1):
            raise ValueError("TRAINING.{} must be {} integer.".format(
                config_name, "a non-negative" if allow_zero else "a positive"))
        return value

    shot = setting("shot", "shot", 1)
    batch_size = setting("batch_size", "train_batch_size", 2)
    workers = setting("num_workers", "num_workers", 1, allow_zero=True)
    train_transform = transform.Compose([
        transform.RandomGaussianBlur(),
        transform.Resize_fy([256, 256]),
        transform.ToTensor(),
        transform.Scaling(),
    ])
    train_data = dataset_us.SemData_US(
        shot=shot, transform=train_transform, mode="train",
        class_dirs=data_params.get("class_dirs"),
        classes=data_params.get("train_classes"),
    )
    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=False,
    )
    print("Training: device=cuda:{}, shot={}, batch_size={}, workers={}, volumes={}".format(
        device, shot, batch_size, workers, len(train_data)))

    # RAP's constructor updates its parameter mapping internally.
    model_params = copy.deepcopy(net_params)
    few_shot_model = fs.RAP(model_params)
    solver = Solver(
        few_shot_model,
        device=device,
        num_class=model_params["num_class"],
        model_name=common_params.get("model_name", "rapnet"),
        exp_name=train_params["exp_name"],
        labels=data_params.get("labels", ["Background", "Foreground"]),
        log_nth=train_params.get("log_nth", 10),
        num_epochs=train_params["num_epochs"],
        use_last_checkpoint=train_params.get("use_last_checkpoint", True),
        log_dir=common_params.get("log_dir", "logs"),
        exp_dir=common_params.get("exp_dir", "experiments"),
    )
    # Solver keeps the release's fixed Adam settings and training-only phase.
    # Its second argument is retained for compatibility, not independent validation.
    solver.train(trainloader, trainloader)


if __name__ == "__main__":
    cfgs, args = get_configs()
    train(cfgs.TRAINING, cfgs.COMMON, cfgs.DATA, cfgs.NETWORK, args)
