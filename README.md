# Processing Pipeline for Heterogeneous 3D Scene Graph Prediction

This repository provides tools for processing the 3DSSG dataset [2] for the publication Renz et al. (2025): Integrating Prior Observations for Incremental 3D Scene Graph Prediction, accepted at ICMLA25.

## Installation

1. Create and activate a conda or pyenv environment (Tested on Python 3.12.2)
2. Navigate to the project directory
3. Install the package: `pip install .`

## Dataset Structure

The dataset expects the following folder structure:

```text
path/to/dataset/root/
├── 3RScan/                    # 3RScan dataset files
├── 3DSSG/                     # 3DSSG annotation files
├── download.py                # Download script from [1]
├── scenegraph.json            # Generated scene graph annotations
├── split_train.json           # Training split
├── split_validate.json        # Validation split
├── split_test.json            # Test split
└── hetero_scene_graph/        # Generated heterogeneous scene graphs
```

## Usage

The main processing pipeline is executed using the `run_3dssg_pipeline.sh` script, which orchestrates the entire data processing workflow. Takes some time.

## Prerequisites

- `download.py` script from [Wald et al. (2019)](https://waldjohannau.github.io/RIO/). (eg. placed at `/path/to/dataset/root`)
- 3RScan dataset with sequences and 3RScan.json (if not using the bash script). Downloading and unpacking takes some time and can also be done using the `download.py` directly
- Sufficient disk space for dataset storage and processing (~300 GB)

### Basic Usage

```bash
./run_3dssg_pipeline.sh --dataset_path /path/to/dataset/root
```

### Command Line Options

The `tools/run_3dssg_pipeline.sh` script supports the following flags:

| Flag | Description | Required | Default |
|------|-------------|----------|---------|
| `--dataset_path` | Path to the dataset root directory | Yes | - |
| `--skip_scans` | Skip downloading 3RScan data | No | false |
| `--download_script` | Path to 3RScan `download.py` (used if scans not skipped) | No | `dataset_path/download.py` |
| `--workers` | Number of workers for frame processing | No | 1 |
| `--scan_id` | Download and process only a specific scan from 3RScan | No | - |
| `--skip_json` | Skip generating `scenegraph.json` and splits | No | false |
| `--skip_embeddings` | Skip generating embeddings | No | false |
| `--numberbatch_path` | Path to numberbatch file or download location | No | `/tmp` |

### Processing Steps

The pipeline consists of several optional processing steps:

1. **Data Download**: Downloads raw dataset files from [1] and scene graph annotation from [2]
2. **JSON Generation**: Creates scene graph annotations and data splits
3. **Rendered Views**: Always generated - creates rendered views of 3D scenes
4. **Heterogeneous Graphs**: Always generated - creates graph representations
5. **Embeddings**: Generates semantic embeddings for objects

### Hierarchical edges

To use the additional hierarchical edges move the `hierarchical` folder and its contents to `path/to/dataset/root`/`hetero_scene_graph`.
The `hetero_scene_graph` subfolder is created by the `generate_hetero_graphs.py` script.

## Train and test models

To train the models we use the pytorch lightning cli:

```python
python -m ssg_tools fit -c path/to/config.yaml
```

To test run

```python
python -m ssg_tools test -c path/to/config.yaml --checkpoint_path path/to/last.ckpt
```

Optionally, you can use the MLFlow logging callback to store artifacts and the config.

Alternatively you find code to run the models in the `getting_started.ipynb`.

## Known issues

There is an issue concerning argparse for Python versions > 3.12.7.
The ofscreen rendering has only been tested successfully on native Ubuntu 22.04, it fails for WSL.

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@inproceedings{renz2025integrating,
  title={Integrating Prior Observations for Incremental 3D Scene Graph Prediction},
  author={Renz, Marian and Igelbrink, Felix and Atzmueller, Martin},
  booktitle={Proceedings of the 24th International Conference on Machine Learning and Applications (ICMLA'25)},
  year={2025},
  organization={IEEE},
  doi={10.48550/arXiv.2509.11895},
  url={https://doi.org/10.48550/arXiv.2509.11895}
}
```

## References

SGFN and PointNet code is based on work from [3].

```text
[1] J. Wald, A. Avetisyan, N. Navab, F. Tombari, and M. Niessner, “RIO: 3D object instance re-localization in changing indoor environments,” in 2019 IEEE/CVF International Conference on Computer Vision (ICCV), Seoul, Korea (South), 2019.

[2] J. Wald, H. Dhamo, N. Navab, and F. Tombari, “Learning 3D semantic scene graphs from 3D indoor reconstructions,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 2020.

[3] S.-C. Wu, J. Wald, K. Tateno, N. Navab, and F. Tombari, “SceneGraphFusion: Incremental 3D scene graph prediction from RGB-D sequences,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Nashville, TN, USA, 2021.
```
