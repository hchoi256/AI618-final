CUDA_VISIBLE_DEVICES=4 python run_tokenflow_pnp.py --config_path configs2/config_pnp_bread.yaml
CUDA_VISIBLE_DEVICES=4 python run_tokenflow_pnp.py --config_path configs2/config_pnp_man-basket.yaml
CUDA_VISIBLE_DEVICES=4 python run_tokenflow_pnp.py --config_path configs2/config_pnp_poodle.yaml

CUDA_VISIBLE_DEVICES=4 python run_tokenflow_pnp_ours.py --config_path configs2/config_pnp_bread.yaml
CUDA_VISIBLE_DEVICES=4 python run_tokenflow_pnp_ours.py --config_path configs2/config_pnp_man-basket.yaml
CUDA_VISIBLE_DEVICES=4 python run_tokenflow_pnp_ours.py --config_path configs2/config_pnp_poodle.yaml