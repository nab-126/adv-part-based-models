# Scaling Part Models: Challenges and Solutions for Robustness on Large Datasets

## Abstract

Part models have been shown to be an effective way to increase the robustness of deep
learning models to adversarial examples. While part models have successfully been applied
to small datasets like PartImageNet, obtaining the necessary segmentation labels can be
expensive and time-consuming. In order to scale part models to larger datasets, it is crucial
to find ways to obtain cheaper labels. In this work, we explore some of the challenges that
may arise when scaling up part models. First, we investigate ways to reduce labeling costs by
using part bounding box labels instead of segmentation masks, while still providing additional
supervision to models. Second, we evaluate the performance of part-based models on a more
diverse and larger dataset. Our work provides valuable insights into the key challenges that
need to be addressed in order to scale up part models successfully and achieve adversarial
robustness on a larger scale.

---

## Setup

### Dependency Installation

- Tested environment can be installed with either `environment.yml` (for `conda`) or `requirements.txt` (for `pip`).
- `python 3.8`
- [`timm`](https://github.com/rwightman/pytorch-image-models)
- [`segmentation-models-pytorch`](https://github.com/qubvel/segmentation_models.pytorch)
- `imagecorruptions` for testing with common corruptions
- `foolbox` for testing with black-box attacks
- `torchmetrics` for computing IoU
- `kornia` for custom transformations

```bash
# Install dependencies with pip
pip install -r requirements.txt

# OR install dependencies with conda
conda env create -f environment.yml

# OR install dependencies manually with latest packages
# Install pytorch 1.10 (or later)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install h5py ruamel.yaml scikit-image timm torchmetrics matplotlib addict yapf pycocotools Cython kornia wandb segmentation-models-pytorch imagecorruptions foolbox termcolor jaxtyping addict yapf tensorboard cmake onnx
pip install black pycodestyle pydocstyle

# To install DINO-related packages (e.g., MultiScaleDeformableAttention)
# MultiScaleDeformableAttention is not compatible with pytorch 2.0 (march 2023)
cd DINO/models/dino/ops
python setup.py build install

# (Optional) install mish cuda
cd ~/temp/ && git clone https://github.com/thomasbrandon/mish-cuda \
    && cd mish-cuda \
    && rm -rf build \
    && mv external/CUDAApplyUtils.cuh csrc/ \
    && python setup.py build install

# For MaskDINO
pip install git+https://github.com/facebookresearch/detectron2.git
pip install git+https://github.com/cocodataset/panopticapi.git
pip install cityscapesscripts

cd MaskDINO
pip install -r requirements.txt
cd maskdino/modeling/pixel_decoder/ops
# If needed to compile on non-cuda node try: TORCH_CUDA_ARCH_LIST='8.0' FORCE_CUDA=1 sh make.sh
sh make.sh
```

### Prepare Part-ImageNet Dataset

```bash
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

# Download data from https://github.com/tacju/partimagenet manually or via gdown
pip install gdown
mkdir ~/data/PartImageNet
gdown https://drive.google.com/drive/folders/1_sKq9g34dsKwfsgLo6j7WCe8Nfceq0Zo -O ~/data/PartImageNet --folder

# Organize the files
cd PartImageNet
unzip train.zip 
unzip test.zip 
unzip val.zip
mkdir JPEGImages
mv train/* test/* val/* JPEGImages
rm -rf train test val train.zip test.zip val.zip
```

### Prepare PACO

Most of the instructions to download PACO can be found [here](https://github.com/facebookresearch/paco). You can also follow the folloing instructions to download PACO annotations with `wget` on a server:

```
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget https://dl.fbaipublicfiles.com/paco/annotations/paco_lvis_v1.zip
wget https://dl.fbaipublicfiles.com/paco/annotations/paco_ego4d_v1.zip
unzip train2017.zip
unzip val2017.zip
unzip paco_lvis_v1.zip
unzip paco_ego4d_v1.zip
rm -rf train2017.zip val2017.zip paco_lvis_v1.zip paco_ego4d_v1.zip
mkdir lvis
mv train2017/ lvis/
mv val2017/ lvis/
```

To download PACO-EGO4D frames, first review and accept the terms of Ego4D usage agreement ([Ego4D Dataset](https://ego4ddataset.com/)). It takes 48 hours to obtain credentials needed for frame download.
Download the frames
```
mkdir tmp
ego4d --output_directory tmp --version v1 --datasets paco_frames
mkdir ego4d
mv tmp/v1/paco_frames/*.jpeg ego4d
```

Finally, please run 
```
sh scripts/prepare_data.sh
```

---

## Scripts

The scripts to reproduce the results in the paper are provided in `scripts/` folder.

## Usage

- Download and prepare the datasets according to the instructions above.
- Run scripts to prepare the datasets for our classification task: `sh scripts/prepare_data.sh`.
- Download weight files from [https://anonfiles.com/1aiaP056y6](https://anonfiles.com/1aiaP056y6).

### Model Naming

We lazily use `--experiment` argument to define the types of model and data to use.
The naming convention is explained below:

```bash
--experiment <SEGTYPE>-<MODELTYPE>[-<MODELARGS>][-no_bg][-semi]
```

- `SEGTYPE`: types of segmentation labels to use
  - `part`: object parts, e.g., dog's legs, cat's legs, dog's head, etc.
  - `object`: objects, e.g., dog, cat, etc.
  - `fg`: background or foreground.
- `MODELTYPE`: types of normal/part models to train/test
  - `wbbox`: Bounding-Box part-based model. Backbone architecture and segmenter architecture are defined by args like `--seg-backbone resnet50 --seg-arch deeplabv3plus`.
  - `pooling`: Downsampled part-based model.
  - `normal`: Normal model (defined by `--arch`, e.g., `--arch resnet50`).


### Example

```[bash]
# Create dir for saving results, logs, and checkpoints
mkdir results
bash scripts/run_example.sh
```

Other tips and tricks

- Set visible GPUs to pytorch (e.g., `CUDA_VISIBLE_DEVICES=0,1,2,3`) if needed.
- Some parameters that might help if NCCL is crashing (maybe for old version of python/pytorch)

```bash
export TORCHELASTIC_MAX_RESTARTS=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
```

## Contributes

- For questions and bug reports, please feel free to create an issue or reach out to us directly.
- Contributions are absolutely welcome.
- Contact: `nabeel126@berkely.edu`

---

## Miscellaneous

Alternative installation via `conda`:

```bash
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -y h5py 
conda install -y -c conda-forge ruamel.yaml scikit-image timm torchmetrics matplotlib addict yapf pycocotools
conda install -y -c anaconda cython
pip install -U kornia wandb segmentation-models-pytorch imagecorruptions foolbox termcolor jaxtyping addict yapf tensorboard cmake onnx
```
