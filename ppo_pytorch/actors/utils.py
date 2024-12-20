import torch
import torch.nn.functional as F
import torch.nn.init as init
from optfn.skip_connections import ResidualBlock
from torch import nn
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
    x = torch.tanh(x * scale).squeeze(1)
    img[:, 0] = img[:, 1] = x.clamp(0, 1)
    img[:, 2] = -x.clamp(-1, 0)
    return img


def weights_init(m, init_alg=init.xavier_uniform_):
    """
    Initialization function for `Actor`. Xavier init is used by default.
    """
    classname = m.__class__.__name__
    conv = classname.find('Conv') != -1
    linear = classname.find('Linear') != -1
    norm = classname.find('Norm') != -1
    layer_2d = classname.find('2d') != -1
    if (conv or linear) and hasattr(m, 'weight'):
        init_alg(m.weight)
        # if m.bias is not None:
        #     m.bias.data.fill_(0)
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
    out = torch.randn(weights.size())
    out *= norm / out.norm(2, dim=1, keepdim=True)
    weights.copy_(out)


def image_to_float(x):
    return x if x.dtype.is_floating_point else x.float().div_(255)


# def multiply_gradient(x: torch.Tensor, mul: float) -> torch.Tensor:
#     x = mul * x
#     x.data /= mul
#     return x


def model_diff(old_model, new_model, max_diff=False) -> float:
    old_state = old_model.state_dict() if hasattr(old_model, 'state_dict') else old_model
    new_state = new_model.state_dict(keep_vars=True) if hasattr(new_model, 'state_dict') else new_model
    norm = 0
    param_count = 0
    for old, new in zip(old_state.values(), new_state.values()):
        if not new.requires_grad or not torch.is_floating_point(new):
            continue
        if max_diff:
            norm = max(norm, (new - old).abs().max().item())
            param_count = 1
        else:
            norm += (new - old).abs().mean().item()
            param_count += 1
    assert param_count != 0
    return norm / param_count


def fixup_init(module):
    with torch.no_grad():
        res_blocks = [m for m in module.modules() if isinstance(m, ResidualBlock)]
        res_block_size = len([m for m in res_blocks[0].modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)])
        assert res_block_size > 0
        weight_mul = len(res_blocks) ** (-1.0 / (2.0 * res_block_size - 2.0))
        for block in res_blocks:
            convs = [m for m in block.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)]
            assert len(convs) == res_block_size
            for i, conv in enumerate(convs):
                mult = 0 if i + 1 == res_block_size else weight_mul
                conv.weight *= mult
                conv.bias *= mult