CUDA_VISIBLE_DEVICES=3 python run_tokenflow_sdedit.py --config_path configs2/config_sdedit_bread.yaml
CUDA_VISIBLE_DEVICES=3 python run_tokenflow_sdedit.py --config_path configs2/config_sdedit_man-basket.yaml
CUDA_VISIBLE_DEVICES=3 python run_tokenflow_sdedit.py --config_path configs2/config_sdedit_poodle.yaml

CUDA_VISIBLE_DEVICES=3 python run_tokenflow_sdedit_ours.py --config_path configs2/config_sdedit_bread.yaml
CUDA_VISIBLE_DEVICES=3 python run_tokenflow_sdedit_ours.py --config_path configs2/config_sdedit_man-basket.yaml
CUDA_VISIBLE_DEVICES=3 python run_tokenflow_sdedit_ours.py --config_path configs2/config_sdedit_poodle.yaml