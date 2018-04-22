import torch
import torch.nn.init as init
from torch.autograd import Variable


def make_conv_heatmap(x, scale=2):
    """
    Convert convolution output to blue-black-yellow color space for better visualization.
    Args:
        x: Conv output.
        scale: Colors clamped to (-scale, scale)
    Returns: Blue-black-yellow images
    """
    img = x.repeat(1, 3, 1, 1).fill_(0)
    x = (x / scale).squeeze(1)
    img[:, 0] = img[:, 1] = x.clamp(0, 1)
    img[:, 2] = -x.clamp(-1, 0)
    return img


def weights_init(m, init_alg=init.xavier_uniform, gain=1):
    """
    Initialization function for `Actor`. Xavier init is used by default.
    """
    classname = m.__class__.__name__
    conv = classname.find('Conv') != -1
    linear = classname.find('Linear') != -1
    norm = classname.find('Norm') != -1
    if (conv or linear) and hasattr(m, 'weight'):
        init_alg(m.weight, gain)
        # if m.bias is not None:
        #     m.bias.data.fill_(0)
    if norm and hasattr(m, 'bias'):
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)


def normalized_columns_initializer(weights, std=1.0):
    """
    Initialization makes layer output have approximately zero mean and `std` standard distribution.
    `std` should be small (0.01) for action probability output in policy gradient to ensure better exploration.
    For state-value or action-value output `std` is usually around 1.
    Args:
        weights: `nn.Linear` weights
        std: Scale of output vector.
    """
    out = torch.randn(weights.size())
    out *= std / out.norm(2, dim=1, keepdim=True)
    weights.copy_(out)


def image_to_float(x):
    name = (x.data if isinstance(x, Variable) else x).__class__.__name__
    return x if name.find('Byte') == -1 else x.float().div_(255)
