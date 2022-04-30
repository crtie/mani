agent = dict(
    type='SAC',
    batch_size=256,
    gamma=0.95,
    update_coeff=0.005,
    alpha=0.2,
    target_update_interval=1,
    automatic_alpha_tuning=True,
    alpha_optim_cfg=dict(type='Adam', lr=0.0003),
    policy_cfg=dict(
        type='ContinuousPolicy',
        policy_head_cfg=dict(
            type='GaussianHead', log_sig_min=-20, log_sig_max=2,
            epsilon=1e-06),
        nn_cfg=dict(
            type='PointNetWithInstanceInfoV0',
            stack_frame=1,
            num_objs='num_objs',
            pcd_pn_cfg=dict(
                type='PointNetV0',
                conv_cfg=dict(
                    type='ConvMLP',
                    norm_cfg=None,
                    mlp_spec=['agent_shape + pcd_xyz_rgb_channel', 192, 192],
                    bias='auto',
                    inactivated_output=True,
                    conv_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
                mlp_cfg=dict(
                    type='LinearMLP',
                    norm_cfg=None,
                    mlp_spec=[192, 192, 192],
                    bias='auto',
                    inactivated_output=True,
                    linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
                subtract_mean_coords=True,
                max_mean_mix_aggregation=True),
            state_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape', 192, 192],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
            transformer_cfg=dict(
                type='TransformerEncoder',
                block_cfg=dict(
                    attention_cfg=dict(
                        type='MultiHeadSelfAttention',
                        embed_dim=192,
                        num_heads=4,
                        latent_dim=32,
                        dropout=0.1),
                    mlp_cfg=dict(
                        type='LinearMLP',
                        norm_cfg=None,
                        mlp_spec=[192, 768, 192],
                        bias='auto',
                        inactivated_output=True,
                        linear_init_cfg=dict(
                            type='xavier_init', gain=1, bias=0)),
                    dropout=0.1),
                pooling_cfg=dict(embed_dim=192, num_heads=4, latent_dim=32),
                mlp_cfg=None,
                num_blocks=2),
            final_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[192, 128, 'action_shape * 2'],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0))),
        optim_cfg=dict(type='Adam', lr=0.0003, weight_decay=5e-06)),
    value_cfg=dict(
        type='ContinuousValue',
        num_heads=2,
        nn_cfg=dict(
            type='PointNetWithInstanceInfoV0',
            stack_frame=1,
            num_objs='num_objs',
            pcd_pn_cfg=dict(
                type='PointNetV0',
                conv_cfg=dict(
                    type='ConvMLP',
                    norm_cfg=None,
                    mlp_spec=[
                        'agent_shape + pcd_xyz_rgb_channel + action_shape',
                        192, 192
                    ],
                    bias='auto',
                    inactivated_output=True,
                    conv_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
                mlp_cfg=dict(
                    type='LinearMLP',
                    norm_cfg=None,
                    mlp_spec=[192, 192, 192],
                    bias='auto',
                    inactivated_output=True,
                    linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
                subtract_mean_coords=True,
                max_mean_mix_aggregation=True),
            state_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + action_shape', 192, 192],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
            transformer_cfg=dict(
                type='TransformerEncoder',
                block_cfg=dict(
                    attention_cfg=dict(
                        type='MultiHeadSelfAttention',
                        embed_dim=192,
                        num_heads=4,
                        latent_dim=32,
                        dropout=0.1),
                    mlp_cfg=dict(
                        type='LinearMLP',
                        norm_cfg=None,
                        mlp_spec=[192, 768, 192],
                        bias='auto',
                        inactivated_output=True,
                        linear_init_cfg=dict(
                            type='xavier_init', gain=1, bias=0)),
                    dropout=0.1),
                pooling_cfg=dict(embed_dim=192, num_heads=4, latent_dim=32),
                mlp_cfg=None,
                num_blocks=2),
            final_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[192, 128, 1],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0))),
        optim_cfg=dict(type='Adam', lr=0.0005, weight_decay=5e-06)))
log_level = 'INFO'
train_mfrl_cfg = dict(
    on_policy=False,
    total_steps=2000000,
    warm_steps=4000,
    n_eval=2000000,
    n_checkpoint=100000,
    n_steps=8,
    n_updates=4)
rollout_cfg = dict(
    type='BatchRollout',
    use_cost=False,
    reward_only=False,
    num_procs=8,
    with_info=False,
    env_cfg=dict(
        type='gym',
        unwrapped=False,
        obs_mode='pointcloud',
        reward_type='dense',
        stack_frame=1,
        env_name='OpenCabinetDrawer_1000-v0'))
eval_cfg = dict(
    type='BatchEvaluation',
    num=100,
    num_procs=8,
    use_hidden_state=False,
    start_state=None,
    save_traj=False,
    save_video=False,
    use_log=True,
    env_cfg=dict(
        type='gym',
        unwrapped=False,
        obs_mode='pointcloud',
        reward_type='dense',
        stack_frame=1,
        env_name='OpenCabinetDrawer_1000-v0'))
stack_frame = 1
num_heads = 4
env_cfg = dict(
    type='gym',
    unwrapped=False,
    obs_mode='pointcloud',
    reward_type='dense',
    stack_frame=1,
    env_name='OpenCabinetDrawer_1000-v0')
expert = dict(
    type='SAC',
    batch_size=1024,
    gamma=0.95,
    policy_cfg=dict(
        type='ContinuousPolicy',
        policy_head_cfg=dict(
            type='GaussianHead', log_sig_min=-20, log_sig_max=2,
            epsilon=1e-06),
        nn_cfg=dict(
            type='PointNetV0',
            conv_cfg=dict(
                type='ConvMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + pcd_all_channel', 256, 512],
                bias='auto',
                inactivated_output=False,
                conv_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
            mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[512, 256, 'action_shape * 2'],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
            subtract_mean_coords=True,
            max_mean_mix_aggregation=True,
            stack_frame=1),
        optim_cfg=dict(type='Adam', lr=0.0005)),
    value_cfg=dict(
        type='ContinuousValue',
        num_heads=2,
        nn_cfg=dict(
            type='PointNetV0',
            conv_cfg=dict(
                type='ConvMLP',
                norm_cfg=None,
                mlp_spec=[
                    'agent_shape + action_shape + pcd_all_channel', 256, 512
                ],
                bias='auto',
                inactivated_output=False,
                conv_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
            mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[512, 256, 1],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
            subtract_mean_coords=True,
            max_mean_mix_aggregation=True,
            stack_frame=1),
        optim_cfg=dict(type='Adam', lr=0.0005)))
replay_cfg = dict(type='ReplayMemory', capacity=800000)
work_dir = './work_dirs/OpenCabinetDrawer_1000-v0/SAC'
