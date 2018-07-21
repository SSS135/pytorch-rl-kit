import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import weight_norm


def make_conv_heatmap(x, scale=0.5):
    """
    Convert convolution output to blue-black-yellow color space for better visualization.
    Args:
        x: Conv output.
        scale: Colors clamped to (-scale, scale)
    Returns: Blue-black-yellow images
    """
    img = x.repeat(1, 3, 1, 1).fill_(0)
    x = F.tanh(x * scale).squeeze(1)
    img[:, 0] = img[:, 1] = x.clamp(0, 1)
    img[:, 2] = -x.clamp(-1, 0)
    return img


def weights_init(m, init_alg=init.xavier_uniform_, gain=1):
    """
    Initialization function for `Actor`. Xavier init is used by default.
    """
    classname = m.__class__.__name__
    conv = classname.find('Conv') != -1
    linear = classname.find('Linear') != -1
    norm = classname.find('Norm') != -1
    layer_2d = classname.find('2d') != -1
    if (conv or linear) and hasattr(m, 'weight'):
        init_alg(m.weight, gain)
        if m.bias is not None:
            m.bias.data.fill_(0)
    # if norm and hasattr(m, 'weight') and m.weight is not None:
    #     m.weight.data.normal_(1, 0.01)
    #     # if layer_2d:
    #     # m.bias.data.normal_(-1, 0.01)


def apply_weight_norm(m, norm=weight_norm):
    """
    Initialization function for `Actor`. Xavier init is used by default.
    """
    classname = m.__class__.__name__
    conv = classname.find('Conv') != -1
    linear = classname.find('Linear') != -1
    if conv or linear:
        print(m)
        norm(m)


def normalized_columns_initializer_(weights, norm=1.0):
    """
    Initialization makes layer output have approximately zero mean and `norm` norm.
    Args:
        weights: `nn.Linear` weights
        norm: Scale of output vector.
    """
    out = torch.nn.init.orthogonal_(weights)
    out *= norm / out.norm(2, dim=1, keepdim=True)
    weights.copy_(out)


def image_to_float(x):
    return x if x.dtype.is_floating_point else x.float().div_(255)
