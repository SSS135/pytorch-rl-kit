import math

import torch

irange = range


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, fill_value=0):
    """
    Same as `torchvision.utils.make_grid`, but with adjustable padding color.

    Given a 4D mini-batch Tensor of shape (B x C x H x W),
    or a list of images all of the same size,
    makes a grid of images of size (B / nrow, nrow).

    normalize=True will shift the image to the range (0, 1),
    by subtracting the minimum and dividing by the maximum pixel value.

    if range=(min, max) where min and max are numbers, then these numbers are used to
    normalize the image.

    scale_each=True will scale each image in the batch of images separately rather than
    computing the (min, max) over all images.

    [Example usage is given in this notebook](https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91)
    """
    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensorlist = tensor
        numImages = len(tensorlist)
        size = torch.Size(torch.Size([numImages]) + tensorlist[0].size())
        tensor = tensorlist[0].new(size)
        for i in irange(numImages):
            tensor[i].copy_(tensorlist[i])

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        return tensor
    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, t.min(), t.max())

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2).fill_(fill_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + 1 + padding // 2, height - padding)\
                .narrow(2, x * width + 1 + padding // 2, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid