python -m tools.run_rl configs/gail/gail_pn.py --seed=0 \
 --cfg-options "env_cfg.env_name=OpenCabinetDrawer-v0" --gpu-ids 0 1 \
 --clean-up \
 --resume-from 'gail_pn.ckpt'