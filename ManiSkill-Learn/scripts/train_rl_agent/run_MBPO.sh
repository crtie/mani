python -m tools.run_rl configs/mbpo/mbpo_mani_skill_state_1M_train.py --seed=0 --cfg-options "env_cfg.env_name=OpenCabinetDrawer_split_train-v0" \
 --gpu-ids 0 1 2 --clean-up --resume-from 'work_dirs/OpenCabinetDrawer_split_train-v0/continue/MBPO/models/model_800000.ckpt' \
 --work-dir 'work_dirs/MBPO/visual'