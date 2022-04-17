import numpy as np
from collections.abc import Sequence, Iterable
from numbers import Number
from .type import str_to_dtype, is_arr, is_dict, is_seq_of, is_type, scalar_type, is_str


def astype(x, dtype):
    if dtype is None:
        return x
    assert is_arr(x) and is_str(dtype), (type(x), type(dtype))
    if is_arr(x, 'np'):
        return x.astype(str_to_dtype(dtype, 'np'))
    elif is_arr(x, 'torch'):
        return x.to(str_to_dtype(dtype, 'torch'))
    elif is_type(dtype):
        return dtype(x)
    else:
        raise NotImplementedError(f"As type {type(x)}")


def to_torch(x, dtype=None, device=None, non_blocking=False):
    import torch
    if x is None:
        return None
    elif is_dict(x):
        return {k: to_torch(x[k], dtype, device, non_blocking) for k in x}
    elif is_seq_of(x):
        return type(x)([to_torch(y, dtype, device, non_blocking) for y in x])

    if isinstance(x, torch.Tensor):
        ret = x.detach()
    elif isinstance(x, (Sequence, Number)):
        ret = torch.from_numpy(np.array(x))
    elif isinstance(x, np.ndarray):
        ret = torch.from_numpy(x)
    else:
        raise NotImplementedError(f"{x} {dtype}")
    if device is not None:
        ret = ret.to(device, non_blocking=non_blocking)
    if dtype is not None:
        ret = astype(ret, dtype)
    return ret


def to_np(x, dtype=None):
    if x is None:
        return None
    elif isinstance(x, str):
        return np.string_(x)
    elif is_dict(x):
        return {k: to_np(x[k], dtype) for k in x}
    elif is_seq_of(x):
        return type(x)([to_np(y, dtype) for y in x])
    elif isinstance(x, (Number, Sequence)):
        ret = np.array(x, dtype=scalar_type(x))
    elif isinstance(x, np.ndarray):
        ret = x
    else:
        import torch
        if isinstance(x, torch.Tensor):
            ret = x.cpu().detach().numpy()
        else:
            raise NotImplementedError(f"{dtype}")
    if dtype is not None:
        ret = astype(ret, dtype)
    return ret


def iter_cast(inputs, dst_type, return_type=None):
    """Cast elements of an iterable object into some type.
    Args:
        inputs (Iterable): The input object.
        dst_type (type): Destination type.
        return_type (type, optional): If specified, the output object will be converted to this type,
                                      otherwise an iterator.
    Returns:
        iterator or specified type: The converted object.
    """
    if not isinstance(inputs, Iterable):
        raise TypeError('inputs must be an iterable object')
    if not isinstance(dst_type, type):
        raise TypeError('"dst_type" must be a valid type')
    out_iterable = map(dst_type, inputs)
    if return_type is None:
        return out_iterable
    else:
        return return_type(out_iterable)


def list_cast(inputs, dst_type):
    return iter_cast(inputs, dst_type, return_type=list)


def tuple_cast(inputs, dst_type):
    return iter_cast(inputs, dst_type, return_type=tuple)


def dict_to_seq(inputs, num_output=2):
    keys = list(sorted(inputs.keys()))
    values = [inputs[k] for k in keys]
    if num_output == 2:
        return keys, values
    elif num_output == 1:
        return tuple(zip(keys, values))
    else:
        raise ValueError(f"num_output is {num_output}, which is not 1 or 2")


def seq_to_dict(*args):
    # args: key, value or a list of list
    args = list(args)
    if len(args) == 2:
        assert len(args[0]) == len(args[1])
        return {args[0][i]: args[1][i] for i in range(len(args[0]))}
    elif len(args) == 1:
        ret = {}
        for item in args:
            assert len(item) == 2
            ret[item[0]] = item[1]
    else:
        raise ValueError(f"len(args) is {len(args)}, which is not 1 or 2")


def dict_to_str(inputs):
    ret = ''
    for key in inputs:
        if ret != '':
            ret += " "
        if isinstance(inputs[key], (float, np.float32, np.float64)):
            if np.abs(inputs[key]).min() < 1E-2:
                ret += f'{key}:{inputs[key]:.4e}'
            else:
                ret += f'{key}:{inputs[key]:.2f}'
        else:
            ret += f'{key}:{inputs[key]}'
    return ret


def number_to_str(x, num):
    if isinstance(x, str):
        return x
    elif np.isscalar(x):
        if np.isreal(x):
            return f'{x:.{num}f}'
        else:
            return str(x)
    else:
        print(type(x))
        raise TypeError(f"Type of {x} is not a number")

def merge_dict(dict1,dict2,permutation):
    # ret={}
    # xyz=np.concatenate((dict1['obs']['pointcloud']['xyz'],dict2['obs']['pointcloud']['xyz']),axis=0)
    # xyz=xyz[permutation]
    # rgb=np.concatenate((dict1['obs']['pointcloud']['rgb'],dict2['obs']['pointcloud']['rgb']),axis=0)
    # rgb=rgb[permutation]
    # seg=np.concatenate((dict1['obs']['pointcloud']['seg'],dict2['obs']['pointcloud']['seg']),axis=0)
    # seg=seg[permutation]
    # state=np.concatenate((dict1['obs']['state'],dict2['obs']['state']),axis=0)
    # state=state[permutation]
    # ret['obs']={'pointcloud':{'xyz':xyz,'rgb':rgb,'seg':seg},'state':state}
    # print(np.concatenate((dict1['actions'],dict2['actions']),axis=0).shape)
    # ret['actions']=np.concatenate((dict1['actions'],dict2['actions']),axis=0)
    # ret['actions']=ret['actions'][permutation]
    # ret['rewards']=np.concatenate((dict1['rewards'],dict2['rewards']),axis=0)
    # ret['rewards']=ret['rewards'][permutation]

    # xyz=np.concatenate((dict1['next_obs']['pointcloud']['xyz'],dict2['next_obs']['pointcloud']['xyz']),axis=0)
    # xyz=xyz[permutation]
    # rgb=np.concatenate((dict1['next_obs']['pointcloud']['rgb'],dict2['next_obs']['pointcloud']['rgb']),axis=0)
    # rgb=rgb[permutation]
    # seg=np.concatenate((dict1['next_obs']['pointcloud']['seg'],dict2['next_obs']['pointcloud']['seg']),axis=0)
    # seg=seg[permutation]
    # state=np.concatenate((dict1['next_obs']['state'],dict2['next_obs']['state']),axis=0)
    # state=state[permutation]
    # ret['next_obs']={'pointcloud':{'xyz':xyz,'rgb':rgb,'seg':seg},'state':state}


    # ret['dones']=np.concatenate((dict1['dones'],dict2['dones']),axis=0)
    # ret['episode_dones']=np.concatenate((dict1['episode_dones'],dict2['episode_dones']),axis=0)


    if is_dict(dict1):
        # print(dict1.keys(),dict2.keys())
        return {k: merge_dict(dict1[k],dict2[k],permutation) for k in dict1.keys()}
    else:
        ret=np.concatenate((dict1,dict2),axis=0)
        ret=ret[permutation]
    return ret