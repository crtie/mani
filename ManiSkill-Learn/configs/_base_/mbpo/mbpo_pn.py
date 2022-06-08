_base_ = ['./mbpo.py']
stack_frame = 1
num_heads = 4

env_cfg = dict(
    type='gym',
    unwrapped=False,
    obs_mode='pointcloud',
    reward_type='dense',
    stack_frame=stack_frame
)



agent = dict(
    type='MBPO',
    batch_size=1024,
    gamma=0.95,
    max_iter_use_real_data=1250,
    use_expert = 1,
    model_updates = 256,
    data_generated_by_model=4, #! 用model造出4倍于env steps的数据
    policy_cfg=dict(
        type='ContinuousPolicy',
        policy_head_cfg=dict(
            type='GaussianHead',
            log_sig_min=-20,
            log_sig_max=2,
            epsilon=1e-6
        ),
        nn_cfg=dict(
            type='PointNetV0',
            conv_cfg=dict(
                type='ConvMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + pcd_all_channel ', 256, 512],
                bias='auto',
                inactivated_output=False,
                conv_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[512 * stack_frame, 256, 'action_shape * 2'],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            subtract_mean_coords=True,
            max_mean_mix_aggregation=True,
            stack_frame=stack_frame,
        ),
        optim_cfg=dict(type='Adam', lr=5e-4),
    ),
    value_cfg=dict(
        type='ContinuousValue',
        num_heads=2,
        nn_cfg=dict(
            type='PointNetV0',
            conv_cfg=dict(
                type='ConvMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + action_shape + pcd_all_channel ', 256, 512],
                bias='auto',
                inactivated_output=False,
                conv_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[512 * stack_frame, 256, 1],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            subtract_mean_coords=True,
            max_mean_mix_aggregation=True,
            stack_frame=stack_frame,
        ),
        optim_cfg=dict(type='Adam', lr=5e-4),
    ),
    #! 照猫画虎写个model
    model_cfg=dict(
        type='Pointnet_model',
        num_heads=1,
        nn_cfg=dict(
            type='PointNetWorldModel',
            conv_cfg=dict(
                type='ConvMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + action_shape + pcd_all_channel ', 256, 512],
                bias='auto',
                inactivated_output=False,
                conv_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[512 * stack_frame, 256, 192],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            #! pooling 出来是一个192维的全局feature
            final_mlp1_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + pcd_all_channel + action_shape + 192', 128, 3],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
            final_mlp2_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + action_shape + 192', 128, 'agent_shape +1 '],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
            subtract_mean_coords=True,
            max_mean_mix_aggregation=True,
            stack_frame=stack_frame,
        ),
        optim_cfg=dict(type='Adam', lr=5e-4, weight_decay=5e-6),
    ),
)


replay_cfg = dict(
    type='ReplayMemory',
    capacity=600000,
)

replay_model_cfg = dict(
    type='ReplayMemory',
    capacity=600000,
)


expert_replay_split_cfg = dict(
    type='ReplayMemory',
    capacity=5000,
)


train_mfrl_cfg = dict( 
    total_steps=2500000,
    warm_steps=4000,
    n_eval=1000,
    n_checkpoint=100000,
    n_steps=8,
    n_updates=8,
    m_steps=8,
    # init_replay_buffers=['option dir/MBPO_baseline/models/buffer.h5',],
    # init_replay_buffers=['option dir/MBPO_baseline/models/buffer.h5',],
)

rollout_cfg = dict(
    type='BatchRollout',
    with_info=False,
    use_cost=False,
    reward_only=False,
    num_procs=8,
)

eval_cfg = dict(
    type='BatchEvaluation',
    num=16,
    num_procs=8,
    use_hidden_state=False,
    start_state=None,
    save_traj=False,
    save_video=False,
    use_log=True,
    env_cfg=dict(
        env_name= "OpenCabinetDrawer_split_validation-v0",
        type='gym',
        unwrapped=False,
        obs_mode='pointcloud',
        reward_type='dense',
        stack_frame=stack_frame
    )
)