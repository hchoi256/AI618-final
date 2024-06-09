# Enhancing Temporal Consistency in Video Editing through Deliberate Keyframe Selection

## Introduction
Drawing inspiration from SLAM technology, we aim to enhance the video editing capabilities of the diffusion model by improving the keyframe selection scheme. Unlike traditional video editing methods that focus solely on the first frame or select keyframes uniformly, we employ a technique akin to SLAM’s global bundle adjustment. Specifically, we add frames to the keyframe list when the differences between images exceed a certain threshold. Our method improved, reducing warp error from 0.1356 to 0.1338 and increasing CLIP score from 0.208 to 0.210 for PnP. Similarly, SDEdit saw enhancements, with warp error decreasing from 0.1788 to 0.1772 and CLIP score rising from 0.2108 to 0.2125

## Installation
### Environment
```bash
conda create -n tokenflow python=3.9 && conda activate tokenflow
pip install -r requirements.txt
```
### Preprocess
```bash
python preprocess.py --data_path data/wolf.mp4 --inversion_prompt ''
python preprocess.py --data_path data/woman-running.mp4 --inversion_prompt ''
```

Download [DAVIS.tar.gz](https://drive.google.com/file/d/174qMDXXp_55A40SkVSK13mUGQGmeLFnq/view?usp=sharing), extract it and place it under the `tokenflow` folder.

## Testing
### SDEdit
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

### PNP
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
