python -m tools.run_rl configs/mbpo/mbpo_mani_skill_state_1M_train.py --seed=0 --cfg-options "env_cfg.env_name=OpenCabinetDrawer_split_train-v0" \
 --gpu-ids 0 --clean-up --work-dir 'work_dirs/test' --resume-from 'work_dirs/new/MBPO_computed_obs/models/model_1200000.ckpt' --evaluation