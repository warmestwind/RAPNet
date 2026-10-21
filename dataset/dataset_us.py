"""NRRD episode loading for the released three-frame RAPNet training path."""

import random
from pathlib import Path

import cv2
import nibabel as nib
import nrrd
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_image(path):
    image = np.asarray(nrrd.read(str(path))[0])
    if image.ndim != 3:
        raise ValueError("Expected a 3-D NRRD image, got {}: {}".format(image.shape, path))
    return image.transpose(2, 1, 0)


def _read_label(path):
    if str(path).endswith(".nii.gz"):
        label = np.asanyarray(nib.load(str(path)).dataobj)
    else:
        label = np.asarray(nrrd.read(str(path))[0])
    if label.ndim != 3:
        raise ValueError("Expected a 3-D label volume, got {}: {}".format(label.shape, path))
    if not np.isfinite(label).all():
        raise ValueError("Label volume contains non-finite values: {}".format(path))
    if np.any(label < 0) or np.any(label > 255) or np.any(label != np.floor(label)):
        raise ValueError("Labels must be integers in [0, 255], with background 0: {}".format(path))
    return label.transpose(2, 1, 0).astype(np.uint8)


def _load_pair(image_path, label_path):
    image, label = _read_image(image_path), _read_label(label_path)
    if image.shape != label.shape:
        raise ValueError("Image/label shape mismatch: {} {} versus {} {}".format(
            image_path, image.shape, label_path, label.shape))
    if image.shape[0] < 3:
        raise ValueError("At least three frames/slices are required, got {}: {}".format(
            image.shape[0], image_path))
    return image, label


def make_dataset(sub_list, class_dirs, shot=1):
    """Index explicit *_im.nrrd / *_m.nrrd (or *_m.nii.gz) pairs.

    Directories are keyed by class/view identifiers such as LV2. Every label
    volume is inspected separately, so a volume is indexed only for labels it
    actually contains. Paths are resolved relative to the working directory.
    """
    if not isinstance(class_dirs, dict) or not class_dirs:
        raise ValueError("Set DATA.class_dirs to a non-empty mapping of class/view names to directories.")
    if not isinstance(sub_list, (list, tuple)) or not sub_list:
        raise ValueError("Set DATA.train_classes to a non-empty list of class/view names.")
    if any(not isinstance(key, str) or not key for key in sub_list):
        raise ValueError("Class/view names must be non-empty strings.")
    if len(set(sub_list)) != len(sub_list):
        raise ValueError("Class/view names must not be repeated.")
    if not isinstance(shot, int) or isinstance(shot, bool) or shot < 1:
        raise ValueError("shot must be a positive integer.")

    image_label_list, sub_class_file_list = [], {}
    for class_name in sub_list:
        if class_name not in class_dirs:
            raise ValueError("No DATA.class_dirs entry for {!r}.".format(class_name))
        root = Path(class_dirs[class_name]).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError("Class/view directory does not exist: {} ({})".format(root, class_name))
        image_paths = sorted(root.glob("*_im.nrrd"))
        if not image_paths:
            raise ValueError("No *_im.nrrd images found in {} ({})".format(root, class_name))

        seen_images = set()
        for image_path in image_paths:
            case_name = image_path.name[:-len("_im.nrrd")]
            candidates = [root / (case_name + suffix) for suffix in ("_m.nrrd", "_m.nii.gz")]
            masks = [path for path in candidates if path.is_file()]
            if len(masks) != 1:
                raise ValueError("Expected exactly one matching mask for {}; found {}. "
                                 "Use {}_m.nrrd or {}_m.nii.gz.".format(
                                     image_path, len(masks), case_name, case_name))
            image_path, label_path = image_path.resolve(), masks[0].resolve()
            if image_path in seen_images:
                raise ValueError("Repeated image volume in class/view {}: {}".format(class_name, image_path))
            seen_images.add(image_path)
            _, label = _load_pair(image_path, label_path)
            labels = [int(value) for value in np.unique(label) if value != 0]
            if not labels:
                raise ValueError("Label volume has no foreground class: {}".format(label_path))

            item = (str(image_path), str(label_path), class_name)
            image_label_list.append(item)
            for label_id in labels:
                key = "{}_{}".format(class_name, label_id)
                sub_class_file_list.setdefault(key, []).append(item)

    for key, items in sub_class_file_list.items():
        distinct_images = {item[0] for item in items}
        if len(distinct_images) < shot + 1:
            raise ValueError("Class/view {} has {} distinct volumes; {} are required "
                             "for one query and {} different support volumes.".format(
                                 key, len(distinct_images), shot + 1, shot))
    return image_label_list, sub_class_file_list


class SemData_US(Dataset):
    def __init__(self, shot=1, transform=None, mode="train", class_dirs=None, classes=None):
        if mode not in ("train", "val", "test"):
            raise ValueError("mode must be train, val, or test.")
        self.mode = mode
        self.shot = shot
        self.sub_list = classes
        self.data_list, self.sub_class_file_list = make_dataset(classes, class_dirs, shot)
        self.transform = transform
        self.slice_num = 3
        self.w_ = 1

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path, class_name = self.data_list[index]
        image, label = _load_pair(image_path, label_path)
        # Preserve uniform random interior-center sampling from the release.
        center = random.randint(1, image.shape[0] - 2)
        image = image[center - 1:center + 2].astype(np.float32)
        label = label[center - 1:center + 2].copy()
        raw_label = label.copy()
        labels = [int(value) for value in np.unique(label) if value != 0]
        if not labels:
            raise ValueError("Sampled query window has no foreground: {} (center {}). "
                             "Use foreground-containing sequences or explicitly adapt the sampling policy.".format(
                                 label_path, center))
        class_chosen = random.choice(labels)
        label = (label == class_chosen).astype(np.uint8)
        chosen_key = "{}_{}".format(class_name, class_chosen)
        eligible = [item for item in self.sub_class_file_list[chosen_key] if item[0] != image_path]
        # A second check makes failures explicit if the index is changed later.
        if len(eligible) < self.shot:
            raise ValueError("Not enough different support volumes for {}: need {}, found {}.".format(
                chosen_key, self.shot, len(eligible)))
        selected = random.sample(eligible, self.shot)

        support_images, support_labels = [], []
        support_image_paths = []
        for support_image_path, support_label_path, _ in selected:
            support_image, support_label = _load_pair(support_image_path, support_label_path)
            support_center = random.randint(1, support_image.shape[0] - 2)
            support_image = support_image[support_center - 1:support_center + 2].astype(np.float32)
            support_label = (support_label[support_center - 1:support_center + 2] == class_chosen).astype(np.uint8)
            if not np.any(support_label):
                raise ValueError("Sampled support window lacks class {}: {} (center {}). "
                                 "Use foreground-containing sequences or explicitly adapt the sampling policy.".format(
                                     class_chosen, support_label_path, support_center))
            support_images.append(support_image)
            support_labels.append(support_label)
            support_image_paths.append(support_image_path)

        # Keep the release's shared left/right flip and per-frame transforms.
        flip = random.random() > 0.5
        if self.transform is not None:
            def apply_window(images, masks):
                transformed_images, transformed_masks = [], []
                for frame, mask in zip(images, masks):
                    if flip:
                        frame, mask = cv2.flip(frame, 1), cv2.flip(mask, 1)
                    frame, mask = self.transform(frame, mask)
                    transformed_images.append(frame)
                    transformed_masks.append(mask.unsqueeze(0))
                return torch.stack(transformed_images), torch.stack(transformed_masks)

            image, label = apply_window(image, label)
            transformed = [apply_window(images, masks)
                           for images, masks in zip(support_images, support_labels)]
            support_images, support_labels = zip(*transformed)
        else:
            # Match the transformed tensor layout without silently scaling data.
            image = torch.from_numpy(image.copy()).unsqueeze(1)
            label = torch.from_numpy(label.astype(np.int64)).unsqueeze(1)
            support_images = [torch.from_numpy(value.copy()).unsqueeze(1) for value in support_images]
            support_labels = [torch.from_numpy(value.astype(np.int64)).unsqueeze(1) for value in support_labels]

        s_xs, s_ys = torch.stack(list(support_images)), torch.stack(list(support_labels))
        subcls_list = [chosen_key] * self.shot
        if self.mode == "train":
            return image, label, s_xs, s_ys, subcls_list
        return image, label, s_xs, s_ys, subcls_list, raw_label, image_path, support_image_paths
