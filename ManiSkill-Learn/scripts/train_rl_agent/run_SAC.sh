python -m tools.run_rl configs/sac/sac_mani_skill_state_1M_train.py --seed=0 --cfg-options "env_cfg.env_name=OpenCabinetDoor_split_no_mid_handle_train-v0" \
 --gpu-ids 0 1 2  --clean-up