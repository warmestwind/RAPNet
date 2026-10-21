# Third-party notices

This document records identified third-party code and existing upstream
attributions in RAPNet. It preserves those notices; it does not select a license
for original RAPNet contributions or establish licensing for the unresolved
components listed below.

The review used RAPNet commit `cf701b7a2c88abdcb87c701809894827d00f5347` as its
baseline. The complete license texts in [`licenses/`](licenses/) were retrieved
from the linked upstream repositories on 2026-10-21 and are retained verbatim.
The upstream links below refer to their `master` branches as reviewed on that
date. The exact historical commits from which RAPNet incorporated the code have
not been established.

## Identified code reuse

| RAPNet code | Upstream source and evidence | Upstream notice and retained license |
| --- | --- | --- |
| `modules/se_modules.py` | [`ai-med/squeeze_and_excitation`](https://github.com/ai-med/squeeze_and_excitation/blob/master/squeeze_and_excitation/squeeze_and_excitation.py): the `ChannelSELayer`, `SpatialSELayer`, `ChannelSpatialSELayer`, and `SELayer` class definitions match the upstream implementation when compared as Python syntax trees. | Copyright (c) 2018 Abhijit Guha Roy. [MIT license](licenses/squeeze-and-excitation-MIT.txt); [upstream license](https://github.com/ai-med/squeeze_and_excitation/blob/master/LICENSE). |
| Parts of `modules/conv_blocks.py` and `losses.py` | [`ai-med/nn-common-modules`](https://github.com/ai-med/nn-common-modules): `ClassifierBlock` matches [`modules.py`](https://github.com/ai-med/nn-common-modules/blob/master/nn_common_modules/modules.py); several other convolution blocks are adapted. `IoULoss`, `CombinedLoss`, `CrossEntropyLoss2d`, `CombinedLoss_KLdiv`, and `FocalLoss` match definitions in [`losses.py`](https://github.com/ai-med/nn-common-modules/blob/master/nn_common_modules/losses.py); `DiceLoss` is modified. | Copyright (c) 2018 Shayan Ahmad Siddiqui. [MIT license](licenses/nn-common-modules-MIT.txt); [upstream license](https://github.com/ai-med/nn-common-modules/blob/master/LICENSE). |
| `FocalLoss` in `losses.py` | The existing source comment explicitly credits [`clcarwin/focal_loss_pytorch`](https://github.com/clcarwin/focal_loss_pytorch). This attribution is retained in addition to the intermediate `nn-common-modules` attribution above. | Copyright (c) 2017 carwin. [MIT license](licenses/focal-loss-pytorch-MIT.txt); [upstream license](https://github.com/clcarwin/focal_loss_pytorch/blob/master/LICENSE). |
| Utility functions in `modules/cre.py` | [`princeton-vl/RAFT`, `core/utils/utils.py`](https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py): `bilinear_sampler` matches the upstream implementation, and `coords_grid` is adapted. This entry describes the checked utility functions, not the provenance of every definition in the file. | Copyright (c) 2020, princeton-vl. All rights reserved. [BSD 3-Clause license](licenses/RAFT-BSD-3-Clause.txt); [upstream license](https://github.com/princeton-vl/RAFT/blob/master/LICENSE). |
| `utils/config.py` and `dataset/transform.py` | [`hszhao/semseg`](https://github.com/hszhao/semseg): `CfgNode` and `_decode_cfg_value` match [`util/config.py`](https://github.com/hszhao/semseg/blob/master/util/config.py). Multiple transform classes match [`util/transform.py`](https://github.com/hszhao/semseg/blob/master/util/transform.py), including `Compose`, `ToTensor`, `Normalize`, `RandScale`, `RandRotate`, and `RandomGaussianBlur`; other routines are adapted. | Copyright (c) 2019 Hengshuang Zhao. [MIT license](licenses/semseg-MIT.txt); [upstream license](https://github.com/hszhao/semseg/blob/master/LICENSE). |

## Existing boundary-loss attribution

`SurfaceLoss` in `losses.py` contains a reference to
[`LIVIAETS/surface-loss`, issue 3](https://github.com/LIVIAETS/surface-loss/issues/3).
The upstream project is now available as
[`LIVIAETS/boundary-loss`](https://github.com/LIVIAETS/boundary-loss).

The upstream [current MIT license](https://github.com/LIVIAETS/boundary-loss/blob/master/LICENSE)
is included verbatim in
[`licenses/boundary-loss-MIT.txt`](licenses/boundary-loss-MIT.txt), with its notice
`Copyright (c) 2023 Hoel Kervadec`. This preserves the available upstream notice;
the precise historical source revision of RAPNet's surface-loss and distance
helper code remains to be confirmed.

## Provenance still to be confirmed

- **`modules/voxelmorph_us.py`:** the file closely matches
  [`zuzhiang/VoxelMorph-torch`, `Model/model.py` at commit
  `a175e142d3226cd6343f51ec3d2e0557ec096c7d`](https://github.com/zuzhiang/VoxelMorph-torch/blob/a175e142d3226cd6343f51ec3d2e0557ec096c7d/Model/model.py).
  The `SpatialTransformer` class and `U_Network.conv_block` / `forward` methods
  match as Python syntax trees. RAPNet changes two input-channel settings from
  two to six and adds a demonstration block. The matched upstream repository
  did not contain a license or notice file at the reviewed revision, so this
  source match does not establish permission to redistribute or relicense it.
  Confirm the applicable terms with the relevant rights holders before claiming
  that a repository-wide license covers this component. The current license of
  a different VoxelMorph repository must not be assumed to cover this copy.
- **Additional legacy utilities and model ancestry:**
  [`abhi4ssj/few-shot-segmentation`](https://github.com/abhi4ssj/few-shot-segmentation)
  is a related upstream project with overlapping file names and structure.
  File-level provenance has not been established for the remaining RAPNet
  utilities. This reference is an item for further review, not an assertion that
  its license applies to every such file.

## Distribution scope

The retained license files apply to the corresponding third-party material.
Keep their copyright notices, license terms, and disclaimers when distributing
that material or substantial portions of it. The RAFT license also contains
conditions for binary distributions and use of contributor names; its complete
text is included above.

External packages, datasets, and pretrained weights have their own terms. They
are not relicensed by this notice file. A license for original RAPNet
contributions requires a separate decision by the relevant rights holders.
