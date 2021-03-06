import torch
import torch.nn as nn
import torch.nn.functional as F
from mani_skill_learn.utils.torch import ExtendedModule
from ..builder import MODELNETWORKS, build_backbone
from ..utils import replace_placeholder_with_args, get_kwargs_from_shape, combine_obs_with_action


@MODELNETWORKS.register_module()
class Ensemble_model(ExtendedModule):
    def __init__(self, nn_cfg, obs_shape=None, action_shape=None, num_heads=1):
        super(Ensemble_model, self).__init__()
        self.values = nn.ModuleList()
        replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
        nn_cfg = replace_placeholder_with_args(nn_cfg, **replaceable_kwargs)
        self.output_dim=obs_shape+1   #! state_size+reward size
        for i in range(num_heads):
            self.values.append(build_backbone(nn_cfg))

    def init_weights(self, pretrained=None, init_cfg=None):
        if not isinstance(pretrained, (tuple, list)):
            pretrained = [pretrained for i in range(len(self.values))]
        for i in range(len(self.values)):
            self.values[i].init_weights(pretrained[i], **init_cfg)

    def forward(self, state, action=None):
        inputs = combine_obs_with_action(state, action)
        ret = [value(inputs) for value in self.values]
        ret=torch.stack(ret,0)
        #! ret of the shape ensemble*batch*(state size+rew size)
        #! 前state+rew 维
        #print(mean.shape,logvar.shape)
        return ret
