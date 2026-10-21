# RAPNet: Learning What and Where to Segment

Official PyTorch implementation of **Learning what and where to segment: A new perspective on medical image few-shot segmentation**, published in *Medical Image Analysis* (2023).

RAPNet combines **registration** and **prototype information** to segment an anatomical target from a small set of annotated support examples. A registration branch transfers spatial information from the support images, while a segmentation branch uses foreground/background prototype similarity. An attention-based fusion module combines these sources of information.

**Paper:** [Medical Image Analysis, 87, 102834](https://doi.org/10.1016/j.media.2023.102834) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/37207524/)

**Authors:** Yong Feng, Yonghuai Wang, Honghe Li, Mingjun Qu, and Jinzhu Yang.

## Contents

- [What is included](#what-is-included)
- [Environment](#environment)
- [Quick execution check](#quick-execution-check)
- [Model interface](#model-interface)
- [Preparing data](#preparing-data)
- [Training](#training)
- [Inference](#inference)
- [Reproducibility and limitations](#reproducibility-and-limitations)
- [Questions and contributions](#questions-and-contributions)
- [Citation](#citation)
- [License](#license)

## What is included

| Path | Purpose |
| --- | --- |
| `RAP.py` | RAPNet model and its `RAP` / `SDnetSegmentor` interfaces |
| `modules/voxelmorph_us.py` | Registration network and spatial transformer |
| `modules/cre.py` | Local feature correlation |
| `modules/conv_blocks.py` | Encoder, decoder, coordinate convolution, and attention blocks |
| `dataset/dataset_us.py` | Episodic sampling of query and support volumes |
| `dataset/transform.py` | Image/mask transforms and intensity scaling |
| `train.py` | Ultrasound training entry point |
| `solver_us_pro.py` | Training loop, loss calculation, and checkpoints |
| `inference.py` | Configurable NRRD inference with recorded support selection |
| `test.py` | Compatibility entry point forwarding to `inference.py` |
| `scripts/smoke_test.py` | Synthetic model forward check with random weights |
| `scripts/make_synthetic_demo.py` | Synthetic NRRD inputs and an untrained checkpoint for an I/O example |
| `configs/ultrasound.example.yaml` | Example data mapping and training configuration |
| `settings.yaml` | Original configuration; also provides the default inference network settings |
| `settings.ini`, `settings.py` | Legacy configuration utilities |
| `requirements-core.txt`, `requirements.txt` | Minimal forward-check and full workflow dependencies |
| `THIRD_PARTY_NOTICES.md`, `licenses/` | Identified upstream sources and retained license notices |
| `cus_loss.py`, `losses.py` | Segmentation losses and related utilities |
| `utils/` | Configuration, logging, preprocessing, and evaluation utilities |

The repository contains the research implementation. **Datasets, pretrained checkpoints, and the original patient split manifests are not included.** The paper reports experiments on echocardiography and abdominal MRI; the supplied training entry point uses the ultrasound-oriented loader described below.

## Environment

Clone the repository and run commands from its root directory:

```bash
git clone https://github.com/warmestwind/RAPNet.git
cd RAPNet
```

Use an isolated Python environment. The maintenance checks below used **Python 3.12.14 and PyTorch 2.7.1+cpu**. The original release listed PyTorch 1.6.0; the modern execution checks do not establish numerical equivalence with that historical environment.

For a CPU-only forward check:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-core.txt
```

For NRRD inference, synthetic file examples, or the training stack, also install:

```bash
python -m pip install -r requirements.txt
```

For GPU training, install the **PyTorch 2.7.1 / torchvision 0.22.1 builds matching your CUDA environment** before installing the requirements. The training loop requires CUDA; the model check and NRRD inference also support CPU. The full requirements use `opencv-python-headless`, which supplies `cv2`, and `pynrrd`, which supplies `nrrd`.

The optional model-summary code invoked directly by `python RAP.py` additionally needs `torchinfo`; the quick check below does not require that package.

## Quick execution check

Run the network with synthetic images, binary support masks, and random weights:

```bash
python scripts/smoke_test.py --device cpu
```

To exercise the multi-support interface:

```bash
python scripts/smoke_test.py --device cpu --shot 2
```

The script checks the three output shapes, finite values, and the probability range, then prints a JSON result. It does not download data or checkpoints.

**Recorded checks on 2026-10-21:**

| Check | Result and scope |
| --- | --- |
| CPU, one and two supports, batch size one | Passed on Python 3.12.14 / PyTorch 2.7.1+cpu. All three outputs had shape `[1, 1, 256, 256]` and finite values. |
| Data loader and training CLI | Fourteen checks with temporary synthetic NRRD/NIfTI fixtures passed, covering explicit image/mask pairing, per-volume class indexing, episode and batch shapes, distinct support selection, CLI parsing, and rejection of invalid inputs. |
| File inference and compatibility | Six checks passed: three-frame NRRD export including empty-foreground support center slices; binary output shape and spatial metadata; repeatable CPU outputs with the same seed; rejection of nonempty outputs without changing files; rejection of conflicting support geometry; integer-image interpolation and CPU DiceLoss compatibility. |

These are software execution checks with synthetic inputs. **No GPU training, clinical-data evaluation, or reproduction of the paper's accuracy was performed in these checks.**

## Model interface

The released model uses **three consecutive grayscale frames/slices** as its image channels and a fixed **256 × 256** registration grid. It produces one foreground segmentation per query center frame/slice.

Let `B` be the batch size and `K` the number of annotated support examples:

| Argument | Shape | Content |
| --- | --- | --- |
| `input1` / query | `[B, 3, 256, 256]` | Previous, center, and next query images |
| `input_sp` / condition | `[K, B, 6, 256, 256]` | Three support image channels followed by three support mask channels, matching the training interface |
| `input_mask` / support masks | `[K, B, 3, 256, 256]` | Binary masks for the three support images |

The model reads the first three condition channels as support images and receives masks separately through `input_mask`. Use floating-point image/mask tensors on the same device as the model. The provided preprocessing divides image values by `255`; adapt this preprocessing deliberately if your images use a different intensity range.

```python
import copy
import torch
from RAP import RAP
from utils.config import load_cfg_from_cfg_file

cfg = load_cfg_from_cfg_file("settings.yaml")
model = RAP(copy.deepcopy(cfg.NETWORK)).eval()

# query, condition, and support_masks must follow the shapes above.
# Move the model and all three tensors to the same device before inference.
with torch.no_grad():
    probability, registered_prior, prototype_prediction = model(
        query, condition, support_masks
    )
```

This interface example assumes that the three input tensors have already been prepared. It does not load data or weights. For trained inference, load a compatible checkpoint before the forward pass.

| Return value | Shape | Meaning |
| --- | --- | --- |
| `probability` | `[B, 1, 256, 256]` | Foreground probability after the final sigmoid |
| `registered_prior` | `[B, 1, 256, 256]` | Mean of the spatially warped support center masks |
| `prototype_prediction` | `[B, 1, 256, 256]` | Upsampled foreground/background decision from a prototype-similarity stage |

The inference command thresholds `probability` at `0.5`. The prototype prediction is a binary decision map, not an uncertainty estimate. A trained checkpoint and appropriately selected annotated support data are needed for meaningful predictions.

## Preparing data

### Volume and label format

`dataset/dataset_us.py` reads image volumes with `nrrd.read` and transposes each array with `.transpose(2, 1, 0)`. After that operation, the loader expects the order `[frames_or_slices, height, width]`. Check array orientation explicitly when exporting your own data; the loader does not infer anatomical orientation from the header.

- Each training volume must have at least three frames/slices, because training samples a center image with one neighbor on either side.
- Image and label volumes must have matching spatial shapes and ordering.
- Background is `0`. The loader selects a nonzero anatomical label for each episode and converts it to a binary foreground mask.
- Labels must be finite integers from `0` to `255`; they are converted to `uint8` after validation.
- The loader has a separate `.nii.gz` label-reading branch. Image reading remains NRRD-based; this is not a general-purpose NIfTI image loader.
- Images are resized with linear interpolation, labels with nearest-neighbor interpolation, and images are divided by `255`.

### Directories and episode construction

Copy `configs/ultrasound.example.yaml` to a configuration for your experiment and set the data directories:

```yaml
DATA:
  class_dirs:
    LV2: data/LV2
    LV3: data/LV3
    LV4: data/LV4
  train_classes: [LV2, LV3, LV4]
  labels: [Background, Foreground]
```

Paths are resolved from the working directory. The identifiers are class/view names chosen for the experiment; use only the directories and classes intended for training. The original patient-level split is not supplied by this example.

Each selected directory should contain pairs such as:

```text
case001_im.nrrd
case001_m.nrrd
case002_im.nrrd
case002_m.nrrd
```

Pairs are matched explicitly by their case prefix. A `case001_m.nii.gz` label can be used instead of `case001_m.nrrd`; supplying both is ambiguous and raises an error. Unrelated files are not treated as image/mask pairs.

The loader checks each volume separately and indexes only the foreground labels that volume contains. For every selected class/view and foreground label, provide at least **`K + 1` distinct eligible volumes**: one query volume and `K` different support volumes. Support selection excludes the query and samples without replacement. Missing masks, shape mismatches, short sequences, empty label volumes, and insufficient support pools raise explicit errors.

The original uniform random interior-frame sampling is retained. Query and support windows are sampled independently. If a sampled query window contains no foreground, or a selected support window lacks the target class, the loader reports the file and center frame instead of inventing a class or repeatedly sampling indefinitely. Use foreground-containing sequences, or explicitly document a different sampling policy when adapting the code.

Separate patients into training, validation, and test sets before building episodes. Patient identity cannot be inferred from a file path, so the loader does not enforce patient-level separation for you. The inherited HDF5 fields in `settings.yaml` are not used by this NRRD loader.

## Training

After setting your data paths, start training with the example configuration:

```bash
python train.py --config configs/ultrasound.example.yaml --device cuda:0
```

The following command overrides episode size, batch size, and data-loader workers:

```bash
python train.py --config configs/ultrasound.example.yaml --device cuda:0 --shot 1 --batch-size 2 --num-workers 0
```

The loader always supplies three-frame windows resized to `256 × 256`. CLI values override their matching YAML values. Device selection respects `CUDA_VISIBLE_DEVICES` set by the caller.

| Configuration | Effect |
| --- | --- |
| `DATA.class_dirs` | Maps class/view names to paired NRRD directories |
| `DATA.train_classes` | Selects the class/view directories used for training |
| `COMMON.device` | CUDA device index or `cuda:N`; overridden by `--device` |
| `TRAINING.shot` | Number of distinct support volumes; overridden by `--shot` |
| `TRAINING.train_batch_size` | Data-loader batch size; overridden by `--batch-size` |
| `TRAINING.num_workers` | Data-loader worker processes; overridden by `--num-workers` |
| `NETWORK` | Model block settings; `RAP.py` sets several channel widths and the binary output internally |
| `TRAINING.num_epochs` | Number of training epochs |
| `TRAINING.exp_name` | Experiment subdirectory name |
| `TRAINING.log_nth` | Iteration logging interval |
| `COMMON.exp_dir`, `COMMON.log_dir` | Root directories for checkpoints and logs |

The model, loss, and optimizer behavior of the research training loop are retained. `solver_us_pro.py` constructs Adam with learning rate `5e-5` and weight decay `1e-5`; inherited YAML optimizer settings do not change those fixed values. A learning-rate scheduler and automatic checkpoint resume are not enabled. `use_last_checkpoint` currently controls preservation of existing log directories, not model-state restoration.

Checkpoints are saved every ten epochs at:

```text
experiments/<experiment_name>/checkpoints/checkpoint_epoch_<epoch>.pth.tar
```

Each checkpoint includes a `state_dict` and optimizer state. There is no additional best-model or final-model save beyond these periodic checkpoints.

The loop performs training only: its second loader argument is retained for compatibility and `Solver.train()` fixes the phase to `train`. **An independent validation and best-checkpoint-selection procedure is not implemented.** Supply a separate validation protocol for new experiments.

## Inference

`inference.py` accepts a checkpoint containing a `state_dict`, or a state dictionary saved directly. Its default network configuration comes from `settings.yaml`; use `--config` when a checkpoint was trained with different compatible settings. Arbitrary pickled model objects are not supported by this loader.

Replace the paths in this example with your own checkpoint and NRRD directories:

```bash
python inference.py \
  --checkpoint path/to/checkpoint_epoch_100.pth.tar \
  --query-dir data/query \
  --support-dir data/support \
  --output-dir outputs/run_001 \
  --device cuda:0 \
  --shot 5 \
  --target-label 1 \
  --seed 0
```

Use `--device cpu` when CUDA is unavailable. `python test.py` accepts the same arguments and forwards to this entry point.

- Query files use `*_im.nrrd` naming. **Query labels are not required for inference.**
- Support files use `*_im.nrrd` with matching `*_m.nrrd` masks. The inference CLI currently expects NRRD support masks.
- Provide at least `--shot` different support volumes after excluding the current query file. Keep support and query patients separate in your evaluation protocol.
- `--target-label` selects the foreground label from support masks. Output masks are binary `0/1`.
- Each query frame uses three neighboring images with boundary replication. Support frames are matched by relative position in their volumes, following the original routine.
- Support sampling uses a local seeded generator. Selected files are recorded for every frame in `run.json`.
- Support image/mask shapes and shared NRRD spatial metadata must agree. The command rejects conflicting metadata and does not resample labels; verify alignment yourself when metadata is absent.
- Support volumes must contain the target label somewhere. A center slice outside the target is allowed, preserving the model's empty-foreground behavior. All-foreground center masks are rejected because the original background prototype has no zero-denominator guard; non-finite predictions also raise an error.
- The output directory must be new or empty. The command does not remove an existing result directory.

For each query volume, the command writes `<case>_pred.nrrd` and `<case>_prior.nrrd`, restores the original array shape, and copies supported NRRD spatial metadata from the query header. `run.json` records the checkpoint path, device, seed, target label, and support choices. This command generates predictions; it does not compute a segmentation-accuracy report.

### Complete synthetic file example

The following creates synthetic NRRD files and an **untrained, randomly initialized checkpoint**, then runs the file-based inference path:

```bash
python scripts/make_synthetic_demo.py --output-dir ../rapnet_demo
python inference.py \
  --checkpoint ../rapnet_demo/random_weights.pth \
  --query-dir ../rapnet_demo/query \
  --support-dir ../rapnet_demo/support \
  --output-dir ../rapnet_demo/predictions \
  --device cpu \
  --shot 1 \
  --seed 0
```

The demo provides two support volumes, so use `--shot 1` or `--shot 2`. Choose a new demo/output directory when repeating the commands. These files exercise loading, tensor construction, prediction export, and run recording. **Their predictions have no demonstrated segmentation accuracy.**

## Reproducibility and limitations

- **Scope:** this is the public research implementation, with an ultrasound-oriented training example. Reproducing the MRI experiments requires the corresponding preprocessing, splits, and evaluation protocol.
- **Assets:** no datasets, trained model weights, original split manifests, or complete historical dependency lockfile are bundled.
- **Validation:** the recorded checks use synthetic data. They establish execution and input/output behavior, not segmentation accuracy, clinical performance, or reproduction of the reported experiments.
- **Resolution:** the registration grid is fixed to `256 × 256`. A different input size requires coordinated model changes.
- **Sampling:** query/support identities, anatomical correspondence, and the intensity scale remain the experimenter's responsibility. Empty sampled windows fail explicitly; the loader does not silently change the sampling policy.
- **Training:** the retained training loop requires CUDA and has fixed Adam settings, periodic checkpoints, no independent validation pass, and no automatic resume. CPU checks cover the model and data interfaces, not GPU optimization.
- **Configuration:** some inherited settings and utility scripts are not connected to the current entry points. The table above identifies the supported training controls.
- **Scientific equivalence:** compatibility fixes and synthetic checks do not establish that a modern environment reproduces every numerical behavior of the original environment.

## Questions and contributions

Please use [GitHub Issues](https://github.com/warmestwind/RAPNet/issues) for questions about setup, data format, model behavior, or reproduction. Include the commit, Python/PyTorch/CUDA versions, command, full traceback, and input tensor shapes where relevant. A minimal synthetic example makes software issues easier to investigate.

Contributions that improve installation, reproducible examples, documentation, portability, and evaluation are welcome. Describe the observed problem, the change, and the checks performed in a pull request. For model or preprocessing changes, distinguish execution checks from experiments demonstrating numerical or scientific equivalence. Do not include patient-identifying data in public issues or pull requests.

## Citation

If this implementation is useful in your research, please cite:

```bibtex
@article{Feng2023RAPNet,
  title   = {Learning what and where to segment: A new perspective on medical image few-shot segmentation},
  author  = {Feng, Yong and Wang, Yonghuai and Li, Honghe and Qu, Mingjun and Yang, Jinzhu},
  journal = {Medical Image Analysis},
  volume  = {87},
  pages   = {102834},
  year    = {2023},
  doi     = {10.1016/j.media.2023.102834}
}
```

## License

Identified third-party sources, retained copyright notices, and full upstream license texts are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/](licenses/). Those notices apply to their corresponding third-party material.

A repository-wide license for the original RAPNet contributions has not yet been selected, and some legacy code provenance remains to be confirmed. Please contact the maintainers through [GitHub Issues](https://github.com/warmestwind/RAPNet/issues) to clarify reuse terms. Preserve the attribution and notices in individual source files.
