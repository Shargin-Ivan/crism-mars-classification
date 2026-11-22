# Mapping minerals on Mars with CRISM hyperspectral data

[Русская версия](README.md)

This student research project was carried out as part of a research training course (Russian: *учебно-исследовательская работа студентов*, УИРС). We tested whether denoising a hyperspectral image improves mineral classification when the models are applied to a different Martian scene.

On the held-out Holden Crater image, the original N2N4M denoiser followed by the Spectral Angle Mapper (SAM) gave the highest accuracy: **87.16% across 20,090 labelled pixels**, compared with **83.85%** for SAM on the original cube. Fine-tuning N2N4M reduced the accuracy of this combination. LWNet on the original cube improved from 24.96% to 30.48% when training increased from 25 to 50 epochs per stage.

[Maps and metrics](results/README.md) · [Data for release v1.0.0](https://github.com/Shargin-Ivan/crism-mars-classification/releases/tag/v1.0.0) · [Training and execution guide in Russian](docs/RUN.md) · [Sources](docs/SOURCES.md)

## Data and processing

CRISM is the Compact Reconnaissance Imaging Spectrometer for Mars. The pipeline reads a complete image cube with its I/F measurements, wavelengths, detector assignments and masks for missing values. SAM and LWNet each classify the original cube. N2N4M runs once, and both classifiers then process the same denoised cube. All four maps share the input spatial grid. The outputs include the class IDs, prediction masks, reasons for rejection, channels used and run configuration.

SAM uses 28 reference spectra from MICA and a fixed spectral-angle threshold of 0.15 radians. The original N2N4M operates on the available measurements within its native 350-channel L-detector grid. Other channels pass through to the output. The fine-tuned models accept each scene's available spectral axis and respect missing measurements and detector boundaries.

| Data | Shape or spectral coverage | Role |
|---|---|---|
| Plebani labelled spectra | 350 spectral channels | First training stage. Training and validation spectra are split by source observation. |
| Nili Fossae, FRT00003E12 | 478 × 593 × 425 | Second training stage, with separate training and validation regions. |
| Utopia Planitia, FRT0000527D | 478 × 595 × 432 | Second training stage, with separate training and validation regions. |
| Holden Crater, FRT0000580C | 418 × 595 × 440 | Held-out scene with six labelled mineral classes. |
| FRT00019468 and HRL000120CD | 1273 × 1296 × 489 and 510 × 370 × 489 | Demonstration scenes without ground-truth maps. |

Cube dimensions are rows × columns × spectral channels. For the two unlabelled scenes, the project reports coverage, rejection rates and agreement with SAM. The map files preserve numeric class IDs and include both the original class names and Russian display labels.

## Results on Holden Crater

Each table entry gives overall accuracy and macro-F1 on the same labelled area. A rejected prediction reduces recall for its reference class. The original LWNet predicts 16 terrestrial land-cover classes from Salinas, so mineral accuracy is undefined for that configuration. Its agreement with other partitions can be examined using the adjusted Rand index and normalized mutual information.

| Processing branch | Original weights | 25 + 25 epochs | 50 + 50 epochs |
|---|---:|---:|---:|
| SAM | 83.85% / 0.7213 | 83.85% / 0.7213 | 83.85% / 0.7213 |
| N2N4M → SAM | **87.16% / 0.7804** | 27.49% / 0.2682 | 18.23% / 0.1809 |
| LWNet | Salinas classes | 24.96% / 0.1501 | **30.48% / 0.2698** |
| N2N4M → LWNet | Salinas classes | 1.20% / 0.0085 | 3.98% / 0.0327 |

N2N4M was fine-tuned by adding synthetic noise to observed spectra and using the observations as reconstruction targets. A lower validation loss for that task does not guarantee better mineral discrimination. The classification accuracy reported here applies to one held-out scene. [All 36 branch results](results/comparison.csv) · [Holden Crater per-class metrics after 50 + 50 epochs](results/epochs50/hymars_hc/per_class_gt.csv)

## Model weights and training

The first epoch count denotes training on Plebani, the second training on Nili Fossae and Utopia Planitia. Each count applies to each model. The two experiments start independently. N2N4M starts from the same published checkpoint in both experiments. The Mars LWNet spectral encoder starts from the same random seed, learns from Plebani spectra and transfers its weights to a spatial model trained on HyMars patches. Both the spectral and spatial components are updated in the second stage.

| Model | Baseline | Best epochs, 25 + 25 | Best epochs, 50 + 50 |
|---|---|---|---|
| LWNet | [Salinas model profile](workspace/configs/models/lwnet_salinas/bundle.json) | [Plebani 24 → HyMars 23](weights/25/lwnet/) | [Plebani 50 → HyMars 49](weights/50/lwnet/) |
| N2N4M | [Published model profile](workspace/configs/models/n2n4m_pretrained/bundle.json) | [Plebani 14 → HyMars 3](weights/25/n2n4m/) | [Plebani 42 → HyMars 4](weights/50/n2n4m/) |

Each stage selects the checkpoint with the lowest validation loss. The final maps use the selected HyMars checkpoints. The 50 + 50 experiment also retains the last model and optimizer states. [Weight paths and SHA-256 checksums](weights/SHA256SUMS)

## Reproduce a map

You can view the 36 published maps directly in the repository. To run inference on Linux, install Git, Git LFS, GitHub CLI and Python 3.11 or 3.12. The GPU setup below uses an NVIDIA driver compatible with the CUDA 12.4 build of PyTorch. Authenticate GitHub CLI with `gh auth login` before downloading the release assets.

```bash
git lfs install
git clone --branch v1.0.0 https://github.com/Shargin-Ivan/crism-mars-classification.git
cd crism-mars-classification
git lfs pull
gh release download v1.0.0 --dir delivery --skip-existing
(cd delivery && sha256sum -c ../docs/DATA_PARTS.sha256)

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
python -m pip install -r requirements.txt
python restore_lwnet.py
python data_parts.py restore delivery
python package_workspace.py verify
python check_weights.py

python infer.py --weights 50 --scene hymars_hc --check
python infer.py --weights 50 --scene hymars_hc
```

The release contains 141 data parts (10.98 GB) and a checksum file. The restore command for LWNet retrieves the pinned upstream Salinas weights and model definition. Inference writes a new run under `outputs/50/hymars_hc/` and prints the HTML report path. Use `--weights standard` or `--weights 25` for the other configurations. The two other scene IDs are `frt00019468_mtrdr` and `hrl000120cd_mtrdr`.

To repeat training on the H100, run `python run_experiment.py run` for the 25 + 25 campaign, followed by `python run_experiment_50.py run` for the independent 50 + 50 campaign. To run the latter directly with the archived baseline for comparison, use `python run_experiment_50.py run --frozen-reference`. On another compatible NVIDIA GPU, set `export CRISM_GPU_PROFILE=portable` before training. [Execution profiles and batch sizes](docs/RUN.md) · [Distributing the data](docs/PUBLISH.md)

## Hardware and measured runtime

The experiments used one NVIDIA H100 NVL with 93.1 GiB of GPU memory visible to PyTorch, PyTorch 2.6.0+cu124 and CUDA runtime 12.4. The rented server was configured with a Xeon Gold 6448Y, 64 allocated CPU cores, 126 GB of advertised system RAM and a 75 GB disk. Peak process memory was not measured.

| Operation on the H100 | Original weights | 25 + 25 epochs | 50 + 50 epochs |
|---|---:|---:|---:|
| Four branches on Holden Crater | 3 min 52 s | 7 min 51 s | 7 min 53 s |
| Four branches on FRT00019468 | 11 min 17 s | 22 min 56 s | 22 min 54 s |
| Four branches on HRL000120CD | 1 min 56 s | 4 min 09 s | 4 min 02 s |
| All three cubes | 17 min 05 s | 34 min 56 s | 34 min 48 s |
| Training both models, both stages | — | 1 h 04 min 16 s | 2 h 07 min 58 s |

Including the baseline runs, training and post-training runs, the first campaign took 1 h 56 min 46 s. The second reused the archived baseline for comparison and took 2 h 43 min 16 s. These timings exclude software installation and data transfer. [Recorded times by step](results/timing.csv)

For a workstation, the suggested starting configuration is one NVIDIA GPU with 24 GB of memory, eight CPU cores, 64 GB of RAM and 100 GB of free SSD space. At an assumed 25–50% of the measured H100 throughput, the three-cube run would take approximately 34–68 minutes with original weights or 70–140 minutes with fine-tuned weights. Training both models would take about 2.1–4.3 hours for 25 + 25 or 4.3–8.5 hours for 50 + 50. These are planning estimates. Measure several warmed-up epochs of each stage and representative image regions to refine them for a particular workstation. [Example 24 GB GPU specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/)

## Sources

- Spectral Angle Mapper by Kruse et al. and the MICA reference library by Viviano-Beck et al. [SAM paper](https://doi.org/10.1016/0034-4257%2893%2990013-N) · [MICA paper](https://doi.org/10.1002/2014JE004627)
- CRISM spectra labelled by Plebani et al. and the HyMars cubes published by Xi et al. [Plebani paper](https://doi.org/10.1016/j.icarus.2021.114849) · [Labelled spectra](https://zenodo.org/records/13338091) · [HyMars V2](https://doi.org/10.57760/sciencedb.19732)
- LWNet by Zhang et al. and N2N4M by Platt, Arcucci and John. [LWNet paper](https://doi.org/10.1109/TGRS.2019.2902568) · [N2N4M paper](https://arxiv.org/abs/2403.17757)
- CRISM observations and metadata from NASA's Planetary Data System. [CRISM archive](https://pds-geosciences.wustl.edu/missions/mro/crism.htm)

The bibliography identifies the publications, authors' implementations and original observations used in this project. [Full source list](docs/SOURCES.md)
