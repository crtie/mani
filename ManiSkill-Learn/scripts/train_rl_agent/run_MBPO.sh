python -m tools.run_rl configs/mbpo/mbpo_mani_skill_state_1M_train.py --seed=0 --cfg-options "env_cfg.env_name=OpenCabinetDrawer_1000_link_0-v0" --num-gpus 1 --clean-up
python -m tools.run_rl configs/mbpo/mbpo_mani_skill_state_1M_train.py --seed=0 --cfg-options "env_cfg.env_name=OpenCabinetDrawer_1000-v0" --num-gpus 1 --clean-up
python -m tools.run_rl configs/mbpo/mbpo_mani_skill_state_1M_train.py --seed=0 --cfg-options "env_cfg.env_name=OpenCabinetDrawer-v0" --num-gpus 2 --clean-up
CUDA_VISIBLE_DEVICES=6,7