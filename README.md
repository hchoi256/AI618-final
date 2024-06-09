# AI 618: Enhancing TokenFlow 

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/aligning-bag-of-regions-for-open-vocabulary/open-vocabulary-object-detection-on-mscoco&#41;]&#40;https://paperswithcode.com/sota/open-vocabulary-object-detection-on-mscoco?p=aligning-bag-of-regions-for-open-vocabulary&#41;)

## Introduction
Recent advancements in text-to-image models have improved image editing, but video editing still struggles with maintaining frame consistency. Current techniques using image diffusion models often lack temporal coherence and rely on selecting keyframes uniformly or from the first frame, which leads to artifacts. To address this, we draw inspiration from SLAM technology, enhancing the keyframe selection scheme of the diffusion model. Instead of traditional methods, we propose selecting keyframes based on the mean square error (MSE) between frames, focusing on those with high differences to improve video continuity.

## Installation

```bash
conda create -n tokenflow python=3.9 && conda activate tokenflow
pip install -r requirements.txt
python preprocess.py --data_path data/wolf.mp4 --inversion_prompt ''
python preprocess.py --data_path data/woman-running.mp4 --inversion_prompt ''
```


## Training and Testing

```bash
python run_tokenflow_sdedit.py --config_path configs/config_sdedit.yaml
python run_tokenflow_sdedit.py --config_path configs/config_sdedit_woman.yaml
python run_tokenflow_sdedit.py --config_path configs2/config_sdedit_bread.yaml
python run_tokenflow_sdedit.py --config_path configs2/config_sdedit_man-basket.yaml
python run_tokenflow_sdedit.py --config_path configs2/config_sdedit_poodle.yaml

python run_tokenflow_sdedit_ours.py --config_path configs/config_sdedit.yaml
python run_tokenflow_sdedit_ours.py --config_path configs/config_sdedit_woman.yaml
python run_tokenflow_sdedit_ours.py --config_path configs2/config_sdedit_bread.yaml
python run_tokenflow_sdedit_ours.py --config_path configs2/config_sdedit_man-basket.yaml
python run_tokenflow_sdedit_ours.py --config_path configs2/config_sdedit_poodle.yaml
```

```bash
python run_tokenflow_pnp.py --config_path configs/config_pnp.yaml
python run_tokenflow_pnp.py --config_path configs/config_pnp_wolf.yaml
python run_tokenflow_pnp.py --config_path configs2/config_pnp_bread.yaml
python run_tokenflow_pnp.py --config_path configs2/config_pnp_man-basket.yaml
python run_tokenflow_pnp.py --config_path configs2/config_pnp_poodle.yaml

python run_tokenflow_pnp_ours.py --config_path configs/config_pnp.yaml
python run_tokenflow_pnp_ours.py --config_path configs/config_pnp_wolf.yaml
python run_tokenflow_pnp_ours.py --config_path configs2/config_pnp_bread.yaml
python run_tokenflow_pnp_ours.py --config_path configs2/config_pnp_man-basket.yaml
python run_tokenflow_pnp_ours.py --config_path configs2/config_pnp_poodle.yaml
```
